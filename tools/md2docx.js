/**
 * Сборка научной статьи из markdown в .docx.
 *
 *   node tools/md2docx.js docs/Статья.md "Статья.docx"
 *
 * Markdown размечается несколькими служебными префиксами, которых нет в
 * обычном markdown, — они несут смысл, не выразимый его синтаксисом:
 *
 *   %АВТОР% / %АФФИЛИАЦИЯ% / %ЗАГОЛОВОК%      — шапка на русском
 *   %АВТОР_EN% / %АФФИЛИАЦИЯ_EN% / %ЗАГОЛОВОК_EN% — то же на английском
 *   %ПОДПИСЬ%  — подпись к таблице (курсив, без отступа)
 *   %ФОРМУЛА%  — выключная формула (по центру, курсив)
 *   %КОД%      — строка листинга моноширинным шрифтом
 *
 * Обычный markdown: ## и ### — заголовки, | … | — таблицы, **жирный**,
 * *курсив*, нумерованные абзацы «1. …» — обычные абзацы.
 *
 * Зависимость: npm-пакет docx. Если он не установлен:  npm install docx
 */
const fs = require("fs");
const path = require("path");

let docx;
try {
  docx = require("docx");
} catch (e) {
  console.error(
    "Не найден пакет docx. Установите его: npm install docx\n" +
      "(либо запускайте скрипт из каталога, где он установлен)"
  );
  process.exit(1);
}
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType,
} = docx;

const FONT = "Times New Roman";
const SIZE = 24; // 12 pt
const SMALL = 22; // 11 pt
const TABLE_W = 9600;

/** Разбирает **жирный** и *курсив* в последовательность TextRun. */
function inline(text, base = {}) {
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      runs.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    }
    const token = m[0];
    if (token.startsWith("**")) {
      runs.push(new TextRun({ text: token.slice(2, -2), font: FONT, bold: true, ...base }));
    } else {
      runs.push(new TextRun({ text: token.slice(1, -1), font: FONT, italics: true, ...base }));
    }
    last = m.index + token.length;
  }
  if (last < text.length) {
    runs.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  }
  return runs.length ? runs : [new TextRun({ text, font: FONT, ...base })];
}

const para = (text, o = {}) =>
  new Paragraph({
    alignment: o.align || AlignmentType.JUSTIFIED,
    spacing: { after: o.after ?? 120, line: 300, before: o.before ?? 0 },
    indent: o.indent === false ? undefined : { firstLine: 567 },
    children: inline(text, { size: o.size || SIZE, bold: o.bold, italics: o.italics }),
  });

function makeTable(rows) {
  const ncols = rows[0].length;
  // Ширина столбца пропорциональна длине самой длинной ячейки, с ограничением.
  const weights = Array.from({ length: ncols }, (_, i) =>
    Math.min(60, Math.max(10, Math.max(...rows.map((r) => (r[i] || "").length))))
  );
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.round((w / total) * TABLE_W));
  // Числовой столбец выравниваем вправо.
  const numeric = Array.from({ length: ncols }, (_, i) =>
    rows.slice(1).every((r) => /^[\d\s.,×%−+-]*$/.test((r[i] || "").trim()) && (r[i] || "").trim())
  );
  return new Table({
    columnWidths: widths,
    width: { size: TABLE_W, type: WidthType.DXA },
    rows: rows.map(
      (cells, ri) =>
        new TableRow({
          tableHeader: ri === 0,
          children: cells.map(
            (c, ci) =>
              new TableCell({
                width: { size: widths[ci], type: WidthType.DXA },
                shading: ri === 0 ? { type: ShadingType.CLEAR, fill: "EFEFEF" } : undefined,
                margins: { top: 60, bottom: 60, left: 100, right: 100 },
                children: [
                  new Paragraph({
                    alignment:
                      ri === 0
                        ? AlignmentType.CENTER
                        : numeric[ci]
                        ? AlignmentType.RIGHT
                        : AlignmentType.LEFT,
                    spacing: { after: 0, line: 260 },
                    children: inline(c, { size: SMALL, bold: ri === 0 }),
                  }),
                ],
              })
          ),
        })
    ),
  });
}

function convert(md) {
  const lines = md.split(/\r?\n/);
  const children = [];
  let i = 0;

  const centeredLine = (text, opts = {}) =>
    children.push(
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: opts.after ?? 100, before: opts.before ?? 0, line: 300 },
        children: inline(text, { size: SIZE, bold: opts.bold }),
      })
    );

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // Служебные префиксы шапки
    let m;
    if ((m = trimmed.match(/^%(АВТОР|АФФИЛИАЦИЯ|АВТОР_EN|АФФИЛИАЦИЯ_EN)%\s*(.*)$/))) {
      centeredLine(m[2], { before: m[1].startsWith("АВТОР") ? 240 : 0 });
      i++;
      continue;
    }
    if ((m = trimmed.match(/^%(ЗАГОЛОВОК|ЗАГОЛОВОК_EN)%\s*(.*)$/))) {
      centeredLine(m[2], { bold: true, before: 240, after: 200 });
      i++;
      continue;
    }
    if ((m = trimmed.match(/^%ПОДПИСЬ%\s*(.*)$/))) {
      children.push(
        new Paragraph({
          spacing: { before: 200, after: 100, line: 300 },
          children: inline(m[1], { size: SIZE, italics: true }),
        })
      );
      i++;
      continue;
    }
    if ((m = trimmed.match(/^%ФОРМУЛА%\s*(.*)$/))) {
      children.push(
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 120, after: 120, line: 300 },
          children: inline(m[1], { size: SIZE, italics: true }),
        })
      );
      i++;
      continue;
    }
    if ((m = trimmed.match(/^%КОД%\s?(.*)$/))) {
      children.push(
        new Paragraph({
          spacing: { after: 0, line: 260 },
          indent: { left: 567 },
          children: [new TextRun({ text: m[1], font: "Courier New", size: 20 })],
        })
      );
      i++;
      continue;
    }

    // Горизонтальная линия — просто разделитель, в документе не нужна
    if (/^-{3,}$/.test(trimmed)) {
      i++;
      continue;
    }

    // Блок кода в тройных кавычках: моноширинный шрифт, строки не склеиваются
    if (trimmed.startsWith("```")) {
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        children.push(
          new Paragraph({
            spacing: { after: 0, line: 260 },
            indent: { left: 567 },
            children: [
              new TextRun({ text: lines[i].replace(/\t/g, "    "), font: "Courier New", size: 20 }),
            ],
          })
        );
        i++;
      }
      i++; // закрывающая ограда
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }

    // Цитата: выключная строка с отступом и курсивом
    if ((m = trimmed.match(/^>\s?(.*)$/))) {
      children.push(
        new Paragraph({
          spacing: { before: 120, after: 120, line: 300 },
          indent: { left: 567 },
          children: inline(m[1], { size: SIZE, italics: true }),
        })
      );
      i++;
      continue;
    }

    // Заголовок документа
    if ((m = trimmed.match(/^#\s+(.*)$/))) {
      children.push(
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 240 },
          children: inline(m[1], { size: 32, bold: true }),
        })
      );
      i++;
      continue;
    }

    // Заголовки
    if ((m = trimmed.match(/^(#{2,3})\s+(.*)$/))) {
      const level = m[1].length;
      children.push(
        new Paragraph({
          heading: level === 2 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
          spacing: { before: level === 2 ? 320 : 240, after: 160 },
          children: inline(m[2], { size: level === 2 ? 26 : SIZE, bold: true }),
        })
      );
      i++;
      continue;
    }

    // Таблица
    if (trimmed.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        const cells = lines[i]
          .trim()
          .replace(/^\||\|$/g, "")
          .split("|")
          .map((c) => c.trim());
        if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) rows.push(cells);
        i++;
      }
      if (rows.length) children.push(makeTable(rows));
      children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
      continue;
    }

    // Обычный абзац (склеиваем последующие непустые строки)
    const buf = [trimmed];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() &&
      !lines[i].trim().startsWith("|") &&
      !lines[i].trim().startsWith("#") &&
      !lines[i].trim().startsWith("%") &&
      // Нумерованный пункт начинает новый абзац: иначе список литературы
      // и перечни в заключении склеились бы в одну простыню.
      !/^\d{1,2}\.\s/.test(lines[i].trim()) &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith(">") &&
      !/^-{3,}$/.test(lines[i].trim())
    ) {
      buf.push(lines[i].trim());
      i++;
    }
    const text = buf.join(" ");
    // Библиография: висячий отступ вместо красной строки
    const isRef = /^\d+\.\s+[А-ЯA-Z]/.test(text) && /—|\/\//.test(text);
    children.push(
      isRef
        ? new Paragraph({
            alignment: AlignmentType.JUSTIFIED,
            spacing: { after: 80, line: 280 },
            indent: { left: 340, hanging: 340 },
            children: inline(text, { size: SMALL }),
          })
        : para(text)
    );
  }
  return children;
}

const [, , input, output] = process.argv;
if (!input || !output) {
  console.error("Использование: node tools/md2docx.js <входной.md> <выходной.docx>");
  process.exit(1);
}
const md = fs.readFileSync(input, "utf8");
const doc = new Document({
  creator: "Симаков В.А.",
  title: path.basename(output, ".docx"),
  styles: { default: { document: { run: { font: FONT, size: SIZE } } } },
  sections: [
    {
      properties: { page: { margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
      children: convert(md),
    },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  try {
    fs.writeFileSync(output, buf);
  } catch (e) {
    if (e.code === "EBUSY" || e.code === "EPERM") {
      console.error(
        `Файл занят другой программой: ${output}\n` +
          "Скорее всего он открыт в Word — закройте его и повторите сборку."
      );
      process.exit(2);
    }
    throw e;
  }
  console.log("собрано:", output);
});

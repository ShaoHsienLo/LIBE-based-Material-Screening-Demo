const MAX_ROWS = 200;

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        field += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      row.push(field);
      field = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') {
        i += 1;
      }
      row.push(field);
      field = '';
      if (row.length > 1 || row[0] !== '') {
        rows.push(row);
      }
      row = [];
      continue;
    }

    field += char;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function createTable(container, headers, rows, total) {
  container.innerHTML = '';
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const tbody = document.createElement('tbody');

  const headRow = document.createElement('tr');
  headers.forEach((h) => {
    const th = document.createElement('th');
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  rows.forEach((row) => {
    const tr = document.createElement('tr');
    headers.forEach((_, idx) => {
      const td = document.createElement('td');
      td.textContent = row[idx] ?? '';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);

  const meta = document.createElement('div');
  meta.className = 'table-meta';
  if (total > rows.length) {
    meta.textContent = `顯示 ${rows.length} / ${total}（最多 ${MAX_ROWS} 筆）`;
  } else {
    meta.textContent = `顯示 ${rows.length} 筆`;
  }
  container.appendChild(meta);

  return { table, tbody };
}

function renderTable(key, tableData, query = '') {
  const { container, headers, rows } = tableData;
  const q = query.trim().toLowerCase();
  const filtered = q
    ? rows.filter((row) => row.some((cell) => (cell || '').toLowerCase().includes(q)))
    : rows;

  const limited = filtered.slice(0, MAX_ROWS);
  createTable(container, headers, limited, filtered.length);
}

function attachSearchHandlers(tables) {
  const inputs = document.querySelectorAll('[data-table-search]');
  inputs.forEach((input) => {
    const key = input.dataset.tableSearch;
    const tableData = tables.get(key);
    if (!tableData) {
      return;
    }
    input.addEventListener('input', (event) => {
      renderTable(key, tableData, event.target.value);
    });
  });
}

async function loadTables() {
  const containers = document.querySelectorAll('.data-table[data-src]');
  const tables = new Map();

  await Promise.all(
    Array.from(containers).map(async (container) => {
      const src = container.dataset.src;
      const key = container.dataset.tableKey;
      try {
        const response = await fetch(src);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const text = await response.text();
        const rows = parseCSV(text);
        if (rows.length === 0) {
          container.textContent = 'CSV 空檔或無資料';
          return;
        }

        const headers = rows[0];
        const dataRows = rows.slice(1);

        const tableData = { container, headers, rows: dataRows };
        tables.set(key, tableData);
        renderTable(key, tableData);
      } catch (err) {
        if (window.location.protocol === 'file:') {
          container.textContent = `載入失敗：${src}（請用本機伺服器或 GitHub Pages 開啟）`;
        } else {
          container.textContent = `載入失敗：${src}`;
        }
      }
    })
  );

  attachSearchHandlers(tables);
}

document.addEventListener('DOMContentLoaded', () => {
  loadTables();
});

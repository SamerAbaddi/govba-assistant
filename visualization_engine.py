import csv
from datetime import datetime
from io import BytesIO, StringIO

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


SUPPORTED_CHART_TYPES = [
    "Bar Chart",
    "Line Chart",
    "Pie Chart",
    "Gantt Chart",
]

MAX_DATA_ROWS = 100
MAX_GANTT_TASKS = 50


def _clean_text(value: object) -> str:
    """Convert a value to clean display text."""

    return str(value).strip()


def _parse_csv_text(data_text: str) -> tuple[list[str], list[list[str]]]:
    """Parse comma-separated text with a header row."""

    if not data_text or not data_text.strip():
        raise ValueError(
            "Structured chart data is required."
        )

    reader = csv.reader(
        StringIO(data_text.strip())
    )

    raw_rows = [
        [_clean_text(cell) for cell in row]
        for row in reader
        if any(_clean_text(cell) for cell in row)
    ]

    if len(raw_rows) < 2:
        raise ValueError(
            "Include a header row and at least one data row."
        )

    headers = raw_rows[0]
    rows = raw_rows[1:]

    if any(not header for header in headers):
        raise ValueError(
            "Every column in the header row must have a name."
        )

    expected_columns = len(headers)

    for row_number, row in enumerate(rows, start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Row {row_number} has {len(row)} columns, "
                f"but the header has {expected_columns}."
            )

    if len(rows) > MAX_DATA_ROWS:
        raise ValueError(
            f"The prototype supports up to {MAX_DATA_ROWS} data rows."
        )

    return headers, rows


def _to_number(value: str, row_number: int) -> float:
    """Convert a text value to a number."""

    cleaned = value.replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError as error:
        raise ValueError(
            f"Value '{value}' in row {row_number} "
            "must be numeric."
        ) from error


def _parse_date(value: str, row_number: int) -> datetime:
    """Parse a date in YYYY-MM-DD format."""

    try:
        return datetime.strptime(
            value.strip(),
            "%Y-%m-%d",
        )
    except ValueError as error:
        raise ValueError(
            f"Date '{value}' in row {row_number} "
            "must use YYYY-MM-DD format."
        ) from error


def _finalize_figure(
    figure: plt.Figure,
) -> bytes:
    """Convert a Matplotlib figure to PNG bytes."""

    output = BytesIO()

    figure.savefig(
        output,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)
    output.seek(0)

    return output.getvalue()


def _create_standard_chart(
    chart_type: str,
    headers: list[str],
    rows: list[list[str]],
    title: str,
    x_label: str,
    y_label: str,
) -> tuple[bytes, dict]:
    """Create a bar, line, or pie chart."""

    if len(headers) < 2:
        raise ValueError(
            f"{chart_type} requires at least two columns."
        )

    categories = [
        row[0]
        for row in rows
    ]

    values = [
        _to_number(row[1], index)
        for index, row in enumerate(rows, start=2)
    ]

    if chart_type == "Pie Chart" and any(
        value < 0 for value in values
    ):
        raise ValueError(
            "Pie-chart values cannot be negative."
        )

    if chart_type == "Pie Chart" and sum(values) <= 0:
        raise ValueError(
            "Pie-chart values must have a total greater than zero."
        )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    if chart_type == "Bar Chart":
        axis.bar(categories, values)
        axis.set_xlabel(
            x_label or headers[0]
        )
        axis.set_ylabel(
            y_label or headers[1]
        )
        axis.tick_params(
            axis="x",
            labelrotation=30,
        )

    elif chart_type == "Line Chart":
        axis.plot(
            categories,
            values,
            marker="o",
        )
        axis.set_xlabel(
            x_label or headers[0]
        )
        axis.set_ylabel(
            y_label or headers[1]
        )
        axis.tick_params(
            axis="x",
            labelrotation=30,
        )
        axis.grid(
            True,
            alpha=0.25,
        )

    else:
        axis.pie(
            values,
            labels=categories,
            autopct="%1.1f%%",
            startangle=90,
        )
        axis.axis("equal")

    axis.set_title(
        title or f"{headers[1]} by {headers[0]}"
    )

    figure.tight_layout()

    metadata = {
        "chart_type": chart_type,
        "row_count": len(rows),
        "category_column": headers[0],
        "value_column": headers[1],
        "title": title or f"{headers[1]} by {headers[0]}",
    }

    return _finalize_figure(figure), metadata


def _create_gantt_chart(
    headers: list[str],
    rows: list[list[str]],
    title: str,
) -> tuple[bytes, dict]:
    """Create a simple Gantt chart."""

    if len(headers) < 3:
        raise ValueError(
            "A Gantt chart requires Task, Start, and End columns."
        )

    if len(rows) > MAX_GANTT_TASKS:
        raise ValueError(
            f"The prototype supports up to "
            f"{MAX_GANTT_TASKS} Gantt tasks."
        )

    tasks = []

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        task_name = row[0]

        if not task_name:
            raise ValueError(
                f"Task name is missing in row {row_number}."
            )

        start_date = _parse_date(
            row[1],
            row_number,
        )
        end_date = _parse_date(
            row[2],
            row_number,
        )

        if end_date < start_date:
            raise ValueError(
                f"The end date in row {row_number} "
                "cannot be earlier than the start date."
            )

        duration_days = (
            end_date - start_date
        ).days + 1

        tasks.append(
            {
                "task": task_name,
                "start": start_date,
                "end": end_date,
                "duration_days": duration_days,
            }
        )

    tasks.sort(
        key=lambda item: item["start"]
    )

    figure_height = max(
        5.5,
        len(tasks) * 0.55 + 2,
    )

    figure, axis = plt.subplots(
        figsize=(11, figure_height)
    )

    task_names = [
        item["task"]
        for item in tasks
    ]

    start_numbers = [
        mdates.date2num(item["start"])
        for item in tasks
    ]

    durations = [
        item["duration_days"]
        for item in tasks
    ]

    y_positions = list(
        range(len(tasks))
    )

    axis.barh(
        y_positions,
        durations,
        left=start_numbers,
    )

    axis.set_yticks(
        y_positions,
        labels=task_names,
    )

    axis.invert_yaxis()
    axis.set_xlabel("Timeline")
    axis.set_title(
        title or "Project Gantt Chart"
    )

    axis.xaxis_date()
    axis.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y-%m-%d")
    )

    axis.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    figure.autofmt_xdate()
    figure.tight_layout()

    metadata = {
        "chart_type": "Gantt Chart",
        "row_count": len(tasks),
        "task_column": headers[0],
        "start_column": headers[1],
        "end_column": headers[2],
        "title": title or "Project Gantt Chart",
    }

    return _finalize_figure(figure), metadata


def create_visualization(
    data_text: str,
    chart_type: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> dict:
    """
    Create a PNG visualization from structured CSV-style text.

    Supported formats:
    - Bar, line, or pie: Category,Value
    - Gantt: Task,Start,End
    """

    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            "Unsupported chart type."
        )

    headers, rows = _parse_csv_text(
        data_text
    )

    if chart_type == "Gantt Chart":
        png_bytes, metadata = _create_gantt_chart(
            headers,
            rows,
            title,
        )

    else:
        png_bytes, metadata = _create_standard_chart(
            chart_type,
            headers,
            rows,
            title,
            x_label,
            y_label,
        )

    return {
        "png_bytes": png_bytes,
        "metadata": metadata,
        "warnings": [
            "The chart reflects only the supplied data.",
            "Verify values, labels, dates, and units before official use.",
        ],
    }
import re

from langchain_groq import ChatGroq

from config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


# =========================================================
# LLM
# =========================================================

def get_llm():
    """
    Initialize and return a ChatGroq LLM instance for plan modification.

    Returns:
        ChatGroq: Configured LLM instance using GPT-OSS-20B model
            with zero temperature and 4096 max tokens for longer outputs.
    """
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )


# =========================================================
# TABLE PARSER
# =========================================================

def parse_markdown_table(plan):
    """
    Parse a markdown table into headers and data rows.

    Args:
        plan (str): Markdown table string.

    Returns:
        tuple: (headers, rows) where:
            - headers (list): List of column header strings.
            - rows (list): List of dicts mapping headers to cell values.
            Returns (None, None) if parsing fails.
    """
    lines = [
        line.strip()
        for line in plan.splitlines()
        if line.strip()
    ]

    table_lines = [
        line
        for line in lines
        if line.startswith("|")
        and line.endswith("|")
    ]

    if len(table_lines) < 3:
        return None, None

    headers = [
        cell.strip()
        for cell in table_lines[0].strip("|").split("|")
    ]

    # Skip separator row.
    data_lines = table_lines[2:]

    rows = []

    for line in data_lines:

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        if len(cells) != len(headers):
            continue

        row = dict(
            zip(headers, cells)
        )

        rows.append(row)

    return headers, rows


# =========================================================
# TABLE BUILDER
# =========================================================

def build_markdown_table(headers, rows):
    """
    Build a markdown table from headers and data rows.

    Args:
        headers (list): List of column header strings.
        rows (list): List of dicts mapping headers to cell values.

    Returns:
        str: Formatted markdown table string.
    """
    output = []

    output.append(
        "| " + " | ".join(headers) + " |"
    )

    output.append(
        "|"
        + "|".join(
            ["---"] * len(headers)
        )
        + "|"
    )

    for row in rows:

        output.append(
            "| "
            + " | ".join(
                row.get(header, "")
                for header in headers
            )
            + " |"
        )

    return "\n".join(output)


# =========================================================
# DAY NUMBER EXTRACTION
# =========================================================

def extract_day_numbers(text):

    matches = re.findall(
        r"\bday\s*(\d+)\b",
        text.lower()
    )

    return [
        int(number)
        for number in matches
    ]


# =========================================================
# DETERMINISTIC MOVE
# =========================================================

def deterministic_move_day(
    current_plan,
    modification
):

    headers, rows = parse_markdown_table(
        current_plan
    )

    if not headers or not rows:
        return None

    required_columns = {
        "Day",
        "Subject",
        "Unit",
        "Topics",
        "Duration",
        "Study Goal"
    }

    if not required_columns.issubset(
        set(headers)
    ):
        return None

    text = modification.lower()

    # Detect requests such as:
    # Move Day 5 to Day 6
    # Move the Day 5 topics to Day 6
    # Shift Day 3 topics to Day 4

    move_match = re.search(
        r"move\s+(?:the\s+)?day\s*(\d+)"
        r"(?:\s+topics?)?"
        r"\s+(?:to|into)\s+day\s*(\d+)",
        text
    )

    if not move_match:
        return None

    source_day = int(
        move_match.group(1)
    )

    target_day = int(
        move_match.group(2)
    )

    source_row = None
    target_row = None

    for row in rows:

        try:
            day = int(
                re.sub(
                    r"[^\d]",
                    "",
                    row["Day"]
                )
            )
        except (ValueError, TypeError):
            continue

        if day == source_day:
            source_row = row

        if day == target_day:
            target_row = row

    if source_row is None or target_row is None:
        return None

    # -----------------------------------------------------
    # If each day has one topic, swap the affected days.
    #
    # This preserves:
    # - daily duration
    # - number of days
    # - all topics
    # - no duplication
    # -----------------------------------------------------

    source_copy = source_row.copy()
    target_copy = target_row.copy()

    for column in headers:

        if column == "Day":
            continue

        source_row[column] = target_copy[column]
        target_row[column] = source_copy[column]

    return build_markdown_table(
        headers,
        rows
    )


# =========================================================
# LLM MODIFICATION
# =========================================================

def llm_modify_study_plan(
    current_plan,
    modification
):

    prompt = f"""
You are a college study-plan modification assistant.

EXISTING STUDY PLAN:

{current_plan}

STUDENT REQUEST:

{modification}

Modify the existing plan according to the request.

STRICT RULES:

1. Preserve every existing topic unless the student
   explicitly asks to remove it.

2. Never invent academic topics.

3. Never duplicate topics.

4. Preserve the number of study days unless explicitly
   asked to change it.

5. Preserve daily study duration unless explicitly
   asked to change it.

6. Preserve:
   - Subject
   - Unit
   - Topics
   - Duration
   - Study Goal

7. Keep EXACTLY this table structure:

| Day | Subject | Unit | Topics | Duration | Study Goal |
|-----|---------|------|--------|----------|------------|

8. Return ONLY the markdown table.

No explanation.
"""

    response = get_llm().invoke(prompt)

    result = response.content

    if isinstance(result, list):

        result = "\n".join(
            str(item)
            for item in result
        )

    return str(result).strip()


# =========================================================
# MAIN MODIFIER
# =========================================================

def modify_study_plan(
    current_plan,
    modification
):

    if not current_plan.strip():

        return (
            "There is no existing study plan to modify."
        )

    if not modification.strip():

        return current_plan

    # -----------------------------------------------------
    # FIRST: deterministic modification
    # -----------------------------------------------------

    deterministic_result = deterministic_move_day(
        current_plan,
        modification
    )

    if deterministic_result:

        return deterministic_result

    # -----------------------------------------------------
    # SECOND: LLM for more complex modifications
    # -----------------------------------------------------

    result = llm_modify_study_plan(
        current_plan,
        modification
    )

    if not result:

        return (
            "The study plan could not be modified. "
            "Please try the modification again."
        )

    # -----------------------------------------------------
    # Validate returned table
    # -----------------------------------------------------

    headers, rows = parse_markdown_table(
        result
    )

    required_columns = {
        "Day",
        "Subject",
        "Unit",
        "Topics",
        "Duration",
        "Study Goal"
    }

    if (
        not headers
        or not rows
        or not required_columns.issubset(
            set(headers)
        )
    ):

        return (
            "The modified study plan had an invalid "
            "format. Please try the modification again."
        )

    return build_markdown_table(
        headers,
        rows
    )

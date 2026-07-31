"""
GovBA Assistant automated smoke tests.

Run:
    python smoke_tests.py

The script checks the six prototype tasks and their main exports.
It does not use real Ministry data, external systems, or paid APIs.
"""

import sys
from dataclasses import dataclass
from typing import Callable

from citizen_qa_engine import answer_citizen_question
from comparison_engine import compare_requirements_documents
from comparison_word_exporter import create_comparison_word_report
from demo_engine import generate_demo_brd
from document_reader import read_uploaded_file
from email_summary_engine import summarize_employee_email
from review_engine import review_requirements_document
from review_word_exporter import create_review_word_report
from visualization_engine import create_visualization
from word_exporter import create_brd_word_report


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


class FakeUploadedFile:
    """Small in-memory uploaded-file substitute for TXT testing."""

    def __init__(
        self,
        name: str,
        content: bytes,
        mime_type: str = "text/plain",
    ) -> None:
        self.name = name
        self.type = mime_type
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content

    def seek(self, position: int) -> None:
        del position


def _require(
    condition: bool,
    message: str,
) -> None:
    """Raise a clear assertion when a test requirement fails."""

    if not condition:
        raise AssertionError(message)


def _is_docx(data: bytes) -> bool:
    """DOCX files are ZIP containers and normally start with PK."""

    return isinstance(data, bytes) and data.startswith(b"PK")


def _is_png(data: bytes) -> bool:
    """Check the standard PNG file signature."""

    return (
        isinstance(data, bytes)
        and data.startswith(b"\x89PNG\r\n\x1a\n")
    )


def test_document_reader() -> None:
    uploaded_file = FakeUploadedFile(
        "sample.txt",
        b"GovBA Assistant sample document.",
    )

    extracted = read_uploaded_file(
        uploaded_file
    )

    _require(
        "GovBA Assistant" in extracted,
        "TXT content was not extracted correctly.",
    )


def test_brd_generation_and_export() -> None:
    source_text = (
        "A government entity wants to automate licence renewal. "
        "Applicants provide a national identification number and "
        "proof of payment. An employee reviews each application. "
        "Approved applicants receive a digital licence."
    )

    result = generate_demo_brd(
        source_text
    )

    required_keys = {
        "service_overview",
        "stakeholders",
        "service_recipients",
        "functional_requirements",
        "non_functional_requirements",
        "business_rules",
        "process_steps",
        "required_data_and_documents",
        "integration_requirements",
        "missing_information",
        "human_review_notes",
    }

    _require(
        isinstance(result, dict),
        "BRD engine did not return a dictionary.",
    )

    _require(
        required_keys.issubset(result.keys()),
        "BRD result is missing one or more required sections.",
    )

    word_report = create_brd_word_report(
        result,
        "Automated smoke-test source",
    )

    _require(
        _is_docx(word_report),
        "BRD Word export is not a valid DOCX byte stream.",
    )


def test_document_review_and_export() -> None:
    document_text = (
        "Service Overview: The service automates licence renewal. "
        "Stakeholders: applicants and licensing employees. "
        "Service Recipients: citizens holding an existing licence. "
        "Functional Requirements: Applicants submit an identification "
        "number and proof of payment. Required Data and Documents: "
        "identification number and payment receipt. An employee should "
        "review requests quickly."
    )

    result = review_requirements_document(
        document_text,
        "BRD",
    )

    _require(
        result.get("document_type") == "BRD",
        "Reviewer returned the wrong document type.",
    )

    completeness = result.get(
        "completeness_indicator"
    )

    _require(
        isinstance(completeness, int)
        and 0 <= completeness <= 100,
        "Reviewer completeness indicator is invalid.",
    )

    _require(
        isinstance(
            result.get("section_checklist"),
            list,
        ),
        "Reviewer section checklist is missing.",
    )

    word_report = create_review_word_report(
        result,
        "Automated smoke-test BRD",
    )

    _require(
        _is_docx(word_report),
        "Review Word export is not a valid DOCX byte stream.",
    )


def test_document_comparison_and_export() -> None:
    document_a = (
        "Applicants must provide an identification number. "
        "Approved applications receive a digital licence within "
        "5 days. The system must notify rejected applicants."
    )

    document_b = (
        "The applicant shall submit an identification number. "
        "Approved requests receive a digital licence within "
        "7 days. The system shall record every completed transaction."
    )

    result = compare_requirements_documents(
        document_a,
        document_b,
        "BRD",
        "SRS",
    )

    _require(
        result.get("document_a_name") == "BRD",
        "Comparison returned the wrong Document A name.",
    )

    _require(
        len(
            result.get(
                "possible_conflicts",
                [],
            )
        ) >= 1,
        "Expected 5-day versus 7-day conflict was not detected.",
    )

    _require(
        len(
            result.get(
                "missing_in_document_b",
                [],
            )
        ) >= 1,
        "Expected missing requirement was not detected.",
    )

    word_report = create_comparison_word_report(
        result
    )

    _require(
        _is_docx(word_report),
        "Comparison Word export is not a valid DOCX byte stream.",
    )


def test_email_summary() -> None:
    email_text = (
        "From: Ahmad Saleh\n"
        "To: Project Team\n"
        "Subject: Urgent review of the service document\n"
        "Please review the attached service requirements and send "
        "your comments by tomorrow. The team agreed to finalize the "
        "document after receiving all comments. Kindly confirm that "
        "the integration section is complete."
    )

    result = summarize_employee_email(
        email_text
    )

    _require(
        result.get("sender") == "Ahmad Saleh",
        "Email sender was not extracted correctly.",
    )

    _require(
        result.get("priority") == "High",
        "Urgent email was not classified as high priority.",
    )

    _require(
        len(result.get("action_items", [])) >= 1,
        "No email action item was detected.",
    )

    _require(
        len(result.get("deadlines", [])) >= 1,
        "No email deadline was detected.",
    )


def test_citizen_question_answering() -> None:
    reference_information = (
        "Licence renewal requires a valid national identification "
        "card and proof of payment. The service fee is 20 JOD. "
        "Processing normally takes three working days."
    )

    supported = answer_citizen_question(
        reference_information,
        "What documents are required for licence renewal?",
    )

    _require(
        supported.get("answer_status")
        == "Supported by supplied information",
        "Supported citizen question was not answered from the source.",
    )

    unsupported = answer_citizen_question(
        reference_information,
        "Can I receive the licence by home delivery?",
    )

    _require(
        unsupported.get("answer_status")
        == "Not found in supplied information",
        "Unsupported citizen question was not safely withheld.",
    )


def test_visualization_creation() -> None:
    bar_result = create_visualization(
        (
            "Department,Requests\n"
            "Licensing,120\n"
            "Payments,85\n"
            "Support,60"
        ),
        "Bar Chart",
        "Service Requests by Department",
    )

    _require(
        _is_png(bar_result.get("png_bytes", b"")),
        "Bar-chart output is not a valid PNG.",
    )

    gantt_result = create_visualization(
        (
            "Task,Start,End\n"
            "Requirements,2026-08-01,2026-08-05\n"
            "Development,2026-08-06,2026-08-15\n"
            "Testing,2026-08-16,2026-08-20"
        ),
        "Gantt Chart",
        "GovBA Prototype Plan",
    )

    _require(
        _is_png(gantt_result.get("png_bytes", b"")),
        "Gantt-chart output is not a valid PNG.",
    )


def run_test(
    name: str,
    test_function: Callable[[], None],
) -> TestResult:
    """Execute one test without stopping the remaining tests."""

    try:
        test_function()

        return TestResult(
            name=name,
            passed=True,
            detail="Passed",
        )

    except Exception as error:
        return TestResult(
            name=name,
            passed=False,
            detail=f"{type(error).__name__}: {error}",
        )


def main() -> int:
    tests = [
        (
            "Document reader",
            test_document_reader,
        ),
        (
            "BRD generation and Word export",
            test_brd_generation_and_export,
        ),
        (
            "BRD review and Word export",
            test_document_review_and_export,
        ),
        (
            "Two-document comparison and Word export",
            test_document_comparison_and_export,
        ),
        (
            "Employee email summary",
            test_email_summary,
        ),
        (
            "Source-grounded citizen Q&A",
            test_citizen_question_answering,
        ),
        (
            "Visualization and PNG export",
            test_visualization_creation,
        ),
    ]

    print("=" * 68)
    print("GovBA Assistant — Automated Smoke Tests")
    print("=" * 68)

    results = [
        run_test(name, function)
        for name, function in tests
    ]

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(
            f"[{status}] {result.name}"
        )

        if not result.passed:
            print(
                f"       {result.detail}"
            )

    passed_count = sum(
        result.passed
        for result in results
    )

    failed_count = len(results) - passed_count

    print("-" * 68)
    print(
        f"Passed: {passed_count} | "
        f"Failed: {failed_count} | "
        f"Total: {len(results)}"
    )

    if failed_count:
        print(
            "Result: One or more tests require correction "
            "before deployment."
        )
        return 1

    print(
        "Result: All automated smoke tests passed. "
        "Proceed to the manual interface checklist."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
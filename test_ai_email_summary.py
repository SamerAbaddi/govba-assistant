"""
Safe AI email-summary fallback test.

This forces AI off and therefore cannot make an external API request.
"""

import os

os.environ["AI_ENABLED"] = "false"

from ai_email_summary_engine import summarize_employee_email_safely


EMAIL = """From: Ahmad Saleh
To: Project Team
Subject: Urgent review of the service document

Please review the attached service requirements and send your comments
by tomorrow. The team agreed to finalize the document after receiving
all comments. Kindly confirm that the integration section is complete.
"""

result = summarize_employee_email_safely(
    EMAIL,
    use_ai=True,
)

assert result["processing_mode"] == "Rule-based fallback"
assert result["fallback_used"] is True
assert result["ai_attempted"] is False
assert result["sender"] == "Ahmad Saleh"
assert result["priority"] == "High"

print("AI email fallback test passed.")
print("No external API request was made.")
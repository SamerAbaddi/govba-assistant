from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


VERSION = "govba-gar-benchmark-v1"

OUT = Path(__file__).with_name("benchmark_manifest.json")

cases = []


def add_pair(
    family,
    index,
    *,
    en,
    ar,
    expected,
    stress=False,
):
    pair_id = f"{family.upper()}-{index:02d}"

    for language, payload in (
        ("en", en),
        ("ar", ar),
    ):
        cases.append(
            {
                "case_id": f"{pair_id}-{language.upper()}",
                "pair_id": pair_id,
                "family": family,
                "language": language,
                "stress": stress,
                **payload,
                "expected": expected,
            }
        )


# ---------------------------------------------------------
# RAG / official-source intelligence
# 10 standard pairs + 5 governance/stress pairs
# ---------------------------------------------------------

rag_standard = [
    (
        "licence renewal",
        "تجديد الترخيص",
        "Applicants renewing a licence must provide identification and proof of payment.",
        "يجب على المتقدمين لتجديد الترخيص تقديم إثبات الهوية وإثبات الدفع.",
    ),
    (
        "building permit",
        "رخصة البناء",
        "A building permit application requires the approved engineering drawings.",
        "يتطلب طلب رخصة البناء المخططات الهندسية المعتمدة.",
    ),
    (
        "business registration",
        "تسجيل المنشأة",
        "Business registration requires identification and the approved registration application.",
        "يتطلب تسجيل المنشأة إثبات الهوية وطلب التسجيل المعتمد.",
    ),
    (
        "vehicle licensing",
        "ترخيص المركبة",
        "Vehicle licensing requires valid insurance and the required inspection record.",
        "يتطلب ترخيص المركبة تأميناً سارياً وسجل الفحص المطلوب.",
    ),
    (
        "passport renewal",
        "تجديد جواز السفر",
        "Passport renewal requires the existing passport and the prescribed application.",
        "يتطلب تجديد جواز السفر جواز السفر الحالي والطلب المقرر.",
    ),
    (
        "social assistance",
        "المساعدة الاجتماعية",
        "Eligibility for social assistance is assessed using the approved applicant information.",
        "يتم تقييم أهلية المساعدة الاجتماعية باستخدام بيانات مقدم الطلب المعتمدة.",
    ),
    (
        "tax clearance",
        "براءة الذمة الضريبية",
        "A tax-clearance request is processed after verifying the applicant's tax record.",
        "تتم معالجة طلب براءة الذمة الضريبية بعد التحقق من السجل الضريبي لمقدم الطلب.",
    ),
    (
        "municipal complaint",
        "الشكوى البلدية",
        "Municipal complaints are registered through the designated complaint channel.",
        "تسجل الشكاوى البلدية من خلال قناة الشكاوى المخصصة.",
    ),
    (
        "scholarship application",
        "طلب المنحة",
        "Scholarship applications require the specified academic supporting documents.",
        "تتطلب طلبات المنح الوثائق الأكاديمية الداعمة المحددة.",
    ),
    (
        "procurement submission",
        "تقديم العطاء",
        "Procurement submissions must include the documents specified in the tender notice.",
        "يجب أن تتضمن العطاءات الوثائق المحددة في إعلان العطاء.",
    ),
]

for i, (en_topic, ar_topic, en_evidence, ar_evidence) in enumerate(
    rag_standard,
    1,
):
    add_pair(
        "rag",
        i,
        en={
            "query": f"What is required for {en_topic}?",
            "evidence": en_evidence,
        },
        ar={
            "query": f"ما المتطلبات اللازمة لـ {ar_topic}؟",
            "evidence": ar_evidence,
        },
        expected={
            "answerability": "ready",
            "trusted": True,
            "temporal_state": "current",
        },
    )


rag_stress = [
    (
        "superseded circular",
        "التعميم الملغى",
        "The earlier circular describes the former application procedure.",
        "يصف التعميم السابق إجراءات التقديم القديمة.",
        "superseded",
    ),
    (
        "conflicting instructions",
        "التعليمات المتعارضة",
        "Two official instructions provide inconsistent submission deadlines.",
        "تتضمن تعليمات رسمية متعارضة مواعيد مختلفة للتقديم.",
        "conflicting",
    ),
    (
        "missing effective date",
        "تاريخ النفاذ المفقود",
        "The document states a procedure but provides no effective date.",
        "تحدد الوثيقة إجراءً ولكنها لا تتضمن تاريخ نفاذ.",
        "insufficient",
    ),
    (
        "untrusted mirror",
        "المصدر غير الموثوق",
        "A copied webpage reproduces government guidance without verified provenance.",
        "تعيد صفحة منسوخة نشر إرشادات حكومية دون مصدر موثق.",
        "untrusted",
    ),
    (
        "insufficient evidence",
        "الأدلة غير الكافية",
        "The available document does not specify the requested eligibility condition.",
        "لا تحدد الوثيقة المتاحة شرط الأهلية المطلوب.",
        "insufficient",
    ),
]

for offset, (
    en_topic,
    ar_topic,
    en_evidence,
    ar_evidence,
    state,
) in enumerate(rag_stress, 11):
    add_pair(
        "rag",
        offset,
        en={
            "query": f"Determine the applicable rule for {en_topic}.",
            "evidence": en_evidence,
        },
        ar={
            "query": f"حدد القاعدة المطبقة بشأن {ar_topic}.",
            "evidence": ar_evidence,
        },
        expected={
            "answerability": (
                "review"
                if state in {"superseded", "conflicting"}
                else "abstain"
            ),
            "trusted": state != "untrusted",
            "temporal_state": state,
        },
        stress=True,
    )


# ---------------------------------------------------------
# Policy / circular change intelligence
# ---------------------------------------------------------

change_pairs = [
    (
        "Employees may submit the form.",
        "Employees must submit the form.",
        "يجوز للموظفين تقديم النموذج.",
        "يجب على الموظفين تقديم النموذج.",
        "MEDIUM",
        "READY",
    ),
    (
        "Applications must be submitted within five days.",
        "Applications must be submitted within three days.",
        "يجب تقديم الطلبات خلال خمسة أيام.",
        "يجب تقديم الطلبات خلال ثلاثة أيام.",
        "MEDIUM",
        "READY",
    ),
    (
        "Applicants may submit documents by email.",
        "Applicants must submit documents through the portal.",
        "يجوز للمتقدمين إرسال الوثائق بالبريد الإلكتروني.",
        "يجب على المتقدمين تقديم الوثائق عبر البوابة.",
        "MEDIUM",
        "READY",
    ),
    (
        "Manager approval is optional.",
        "Manager approval is required.",
        "موافقة المدير اختيارية.",
        "موافقة المدير مطلوبة.",
        "HIGH",
        "READY",
    ),
    (
        "The department should review the request.",
        "The department must review the request.",
        "ينبغي للدائرة مراجعة الطلب.",
        "يجب على الدائرة مراجعة الطلب.",
        "MEDIUM",
        "READY",
    ),
    (
        "Records are retained for three years.",
        "Records are retained for five years.",
        "تحفظ السجلات لمدة ثلاث سنوات.",
        "تحفظ السجلات لمدة خمس سنوات.",
        "MEDIUM",
        "READY",
    ),
    (
        "Payment may be completed manually.",
        "Payment must be completed electronically.",
        "يجوز إتمام الدفع يدوياً.",
        "يجب إتمام الدفع إلكترونياً.",
        "MEDIUM",
        "READY",
    ),
    (
        "The report is submitted quarterly.",
        "The report is submitted monthly.",
        "يقدم التقرير كل ثلاثة أشهر.",
        "يقدم التقرير شهرياً.",
        "LOW",
        "READY",
    ),
    (
        "Supporting documents are recommended.",
        "Supporting documents are mandatory.",
        "يوصى بإرفاق الوثائق الداعمة.",
        "الوثائق الداعمة إلزامية.",
        "MEDIUM",
        "READY",
    ),
    (
        "The request may be reviewed remotely.",
        "The request must be reviewed through the approved platform.",
        "يجوز مراجعة الطلب عن بعد.",
        "يجب مراجعة الطلب من خلال المنصة المعتمدة.",
        "MEDIUM",
        "READY",
    ),
]

for i, (
    en_old,
    en_new,
    ar_old,
    ar_new,
    impact,
    briefing,
) in enumerate(change_pairs, 1):
    add_pair(
        "change",
        i,
        en={
            "old": [en_old],
            "new": [en_new],
        },
        ar={
            "old": [ar_old],
            "new": [ar_new],
        },
        expected={
            "match": "ALLOW",
            "added": 0,
            "removed": 0,
            "modified": 1,
            "unchanged": 0,
            "impact": impact,
            "briefing": briefing,
        },
    )


change_stress = [
    {
        "en_old": ["Applications are submitted online."],
        "en_new": [
            "Applications are submitted online.",
            "Applicants must attach identification.",
        ],
        "ar_old": ["تقدم الطلبات إلكترونياً."],
        "ar_new": [
            "تقدم الطلبات إلكترونياً.",
            "يجب على المتقدم إرفاق إثبات الهوية.",
        ],
        "expected": {
            "match": "ALLOW",
            "added": 1,
            "removed": 0,
            "modified": 0,
            "unchanged": 1,
            "impact": "MEDIUM",
            "briefing": "READY",
        },
    },
    {
        "en_old": [
            "Applications are submitted online.",
            "A paper copy is also required.",
        ],
        "en_new": ["Applications are submitted online."],
        "ar_old": [
            "تقدم الطلبات إلكترونياً.",
            "تطلب نسخة ورقية أيضاً.",
        ],
        "ar_new": ["تقدم الطلبات إلكترونياً."],
        "expected": {
            "match": "ALLOW",
            "added": 0,
            "removed": 1,
            "modified": 0,
            "unchanged": 1,
            "impact": "MEDIUM",
            "briefing": "READY",
        },
    },
    {
        "en_old": ["The approved procedure remains in force."],
        "en_new": ["The approved procedure remains in force."],
        "ar_old": ["يبقى الإجراء المعتمد نافذاً."],
        "ar_new": ["يبقى الإجراء المعتمد نافذاً."],
        "expected": {
            "match": "ALLOW",
            "added": 0,
            "removed": 0,
            "modified": 0,
            "unchanged": 1,
            "impact": "INFORMATIONAL",
            "briefing": "READY",
        },
    },
    {
        "en_old": ["Sensitive records may be stored without encryption."],
        "en_new": ["Sensitive records must be encrypted at rest."],
        "ar_old": ["يجوز حفظ السجلات الحساسة دون تشفير."],
        "ar_new": ["يجب تشفير السجلات الحساسة أثناء التخزين."],
        "expected": {
            "match": "ALLOW",
            "added": 0,
            "removed": 0,
            "modified": 1,
            "unchanged": 0,
            "impact": "CRITICAL",
            "briefing": "REVIEW",
        },
    },
    {
        "en_old": ["The deadline is 30 September."],
        "en_new": ["The deadline is 15 September."],
        "ar_old": ["الموعد النهائي هو 30 أيلول."],
        "ar_new": ["الموعد النهائي هو 15 أيلول."],
        "expected": {
            "match": "ALLOW",
            "added": 0,
            "removed": 0,
            "modified": 1,
            "unchanged": 0,
            "impact": "HIGH",
            "briefing": "READY",
        },
    },
]

for i, item in enumerate(change_stress, 11):
    add_pair(
        "change",
        i,
        en={
            "old": item["en_old"],
            "new": item["en_new"],
        },
        ar={
            "old": item["ar_old"],
            "new": item["ar_new"],
        },
        expected=item["expected"],
        stress=True,
    )


# ---------------------------------------------------------
# Correspondence intelligence
# ---------------------------------------------------------

correspondence = [
    (
        "Please review the attached report.",
        "يرجى مراجعة التقرير المرفق.",
        "REQUEST", "NORMAL", 1, 0, 0, "READY",
    ),
    (
        "Urgent: please submit the report.",
        "عاجل: يرجى تقديم التقرير.",
        "REQUEST", "URGENT", 1, 0, 0, "REVIEW",
    ),
    (
        "For your information, the report has been issued.",
        "للعلم، تم إصدار التقرير.",
        "INFORMATION", "ROUTINE", 0, 0, 0, "READY",
    ),
    (
        "Please approve the proposed service procedure.",
        "يرجى الموافقة على إجراء الخدمة المقترح.",
        "APPROVAL", "NORMAL", 1, 0, 0, "READY",
    ),
    (
        "The committee has decided to approve the request.",
        "قررت اللجنة الموافقة على الطلب.",
        "DECISION", "NORMAL", 0, 0, 0, "READY",
    ),
    (
        "Please escalate this issue to senior management.",
        "يرجى تصعيد هذه المسألة إلى الإدارة العليا.",
        "ESCALATION", "HIGH", 1, 0, 0, "REVIEW",
    ),
    (
        "I wish to complain about the delayed service.",
        "أرغب في تقديم شكوى بشأن تأخر الخدمة.",
        "COMPLAINT", "HIGH", 0, 0, 0, "REVIEW",
    ),
    (
        "Please note that the portal will be unavailable tomorrow.",
        "يرجى العلم أن البوابة ستكون غير متاحة غداً.",
        "NOTIFICATION", "ROUTINE", 0, 0, 0, "READY",
    ),
    (
        "You must submit the signed form.",
        "يجب عليك تقديم النموذج الموقع.",
        "ACTION_REQUIRED", "NORMAL", 1, 0, 0, "READY",
    ),
    (
        "Please provide your comments on the draft.",
        "يرجى تزويدنا بملاحظاتكم على المسودة.",
        "REQUEST", "NORMAL", 1, 0, 0, "READY",
    ),
    (
        "Immediate action is required to resolve the security issue.",
        "يلزم اتخاذ إجراء فوري لمعالجة المسألة الأمنية.",
        "ACTION_REQUIRED", "URGENT", 1, 0, 0, "REVIEW",
    ),
    (
        "We will submit the final report.",
        "سنقدم التقرير النهائي.",
        "INFORMATION", "NORMAL", 1, 1, 0, "READY",
    ),
    (
        "Please submit the response by tomorrow.",
        "يرجى تقديم الرد بحلول الغد.",
        "REQUEST", "HIGH", 1, 0, 1, "REVIEW",
    ),
    (
        "The citizen alleges repeated failure to process the request.",
        "يشكو المواطن من تكرار عدم معالجة الطلب.",
        "COMPLAINT", "HIGH", 0, 0, 0, "REVIEW",
    ),
    (
        "Escalate immediately: the service interruption affects all users.",
        "صعّد الموضوع فوراً: انقطاع الخدمة يؤثر على جميع المستخدمين.",
        "ESCALATION", "URGENT", 1, 0, 0, "REVIEW",
    ),
]

for i, (
    en_text,
    ar_text,
    intent,
    priority,
    actions,
    commitments,
    deadlines,
    briefing,
) in enumerate(correspondence, 1):
    add_pair(
        "correspondence",
        i,
        en={"text": en_text},
        ar={"text": ar_text},
        expected={
            "intent": intent,
            "priority": priority,
            "actions": actions,
            "commitments": commitments,
            "deadlines": deadlines,
            "briefing": briefing,
        },
        stress=i > 10,
    )


# ---------------------------------------------------------
# Requirements / BRD intelligence
# ---------------------------------------------------------

requirements = [
    (
        "The system must allow users to submit requests.",
        "يجب أن يسمح النظام للمستخدمين بتقديم الطلبات.",
        "FUNCTIONAL", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system must use encryption.",
        "يجب أن يستخدم النظام التشفير.",
        "SECURITY", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system should generate monthly reports.",
        "ينبغي أن ينشئ النظام تقارير شهرية.",
        "REPORTING", "SHOULD", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system must store the applicant reference number.",
        "يجب أن يخزن النظام الرقم المرجعي لمقدم الطلب.",
        "DATA", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system must integrate with the payment API.",
        "يجب أن يتكامل النظام مع واجهة برمجة تطبيقات الدفع.",
        "INTEGRATION", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The solution must comply with the applicable policy.",
        "يجب أن يمتثل الحل للسياسة المعمول بها.",
        "COMPLIANCE", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The service must reduce manual processing.",
        "يجب أن تقلل الخدمة المعالجة اليدوية.",
        "BUSINESS", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system should respond within two seconds.",
        "ينبغي أن يستجيب النظام خلال ثانيتين.",
        "NON_FUNCTIONAL", "SHOULD", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system could provide status notifications.",
        "يمكن للنظام توفير إشعارات عن حالة الطلب.",
        "FUNCTIONAL", "COULD", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system must provide an audit record.",
        "يجب أن يوفر النظام سجل تدقيق.",
        "SECURITY", "MUST", "DRAFT_REVIEW", "PASS", "REVIEW",
    ),
    (
        "The system should provide a user-friendly interface.",
        "ينبغي أن يوفر النظام واجهة سهلة الاستخدام.",
        "FUNCTIONAL", "SHOULD", "DRAFT_REVIEW", "REVIEW", "REVIEW",
    ),
    (
        "The system must support TBD response time.",
        "يجب أن يدعم النظام زمن استجابة يحدد لاحقاً.",
        "NON_FUNCTIONAL", "MUST", "DRAFT_REVIEW", "REWRITE", "REVIEW",
    ),
    (
        "The system should be fast.",
        "ينبغي أن يكون النظام سريعاً.",
        "NON_FUNCTIONAL", "SHOULD", "DRAFT_REVIEW", "REWRITE", "REVIEW",
    ),
    (
        "The system must provide appropriate notifications.",
        "يجب أن يوفر النظام إشعارات مناسبة.",
        "FUNCTIONAL", "MUST", "DRAFT_REVIEW", "REVIEW", "REVIEW",
    ),
    (
        "The system will not provide paper-based submission.",
        "لن يوفر النظام التقديم الورقي.",
        "FUNCTIONAL", "WONT", "NOT_APPLICABLE", "PASS", "READY",
    ),
]

for i, (
    en_text,
    ar_text,
    req_type,
    priority,
    acceptance,
    quality,
    decision,
) in enumerate(requirements, 1):
    add_pair(
        "requirements",
        i,
        en={"text": en_text},
        ar={"text": ar_text},
        expected={
            "types": [req_type],
            "priorities": [priority],
            "acceptance": [acceptance],
            "quality": [quality],
            "decision": decision,
        },
        stress=i > 10,
    )


# ---------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------

assert len(cases) == 120

family_counts = Counter(case["family"] for case in cases)
language_counts = Counter(case["language"] for case in cases)

assert family_counts == {
    "rag": 30,
    "change": 30,
    "correspondence": 30,
    "requirements": 30,
}

assert language_counts == {
    "en": 60,
    "ar": 60,
}

assert len({case["case_id"] for case in cases}) == 120

for case in cases:
    if case["family"] == "correspondence":
        expected = case["expected"]

        if expected["commitments"] > expected["actions"]:
            raise AssertionError(
                f"{case['case_id']}: commitments exceed actions"
            )

        if expected["deadlines"] > expected["actions"]:
            raise AssertionError(
                f"{case['case_id']}: deadlines exceed actions"
            )

manifest = {
    "version": VERSION,
    "case_count": len(cases),
    "family_counts": dict(family_counts),
    "language_counts": dict(language_counts),
    "cases": cases,
}

OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print("benchmark_version:", VERSION)
print("case_count:", len(cases))
print("family_counts:", dict(family_counts))
print("language_counts:", dict(language_counts))
print("output:", OUT)
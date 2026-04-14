import unittest

from curriculum_matcher.url_evidence import (
    audit_url_field,
    classify_url,
    extract_title_from_html,
    normalize_evidence_text,
)


class UrlEvidenceTests(unittest.TestCase):
    def test_classify_url_detects_common_types(self):
        self.assertEqual(classify_url("https://docs.google.com/document/d/abc/edit"), "google_doc")
        self.assertEqual(classify_url("https://example.org/files/board-agenda.pdf"), "pdf")
        self.assertEqual(classify_url("https://www.hmhco.com/programs/example"), "vendor_page")
        self.assertEqual(classify_url("https://district.k12.ca.us/adoption"), "district_or_org_page")

    def test_extract_title_from_html(self):
        html = "<html><head><title>Sample Curriculum Page</title></head><body>Body</body></html>"
        self.assertEqual(extract_title_from_html(html), "Sample Curriculum Page")

    def test_audit_url_field_missing_without_fetch(self):
        audit = audit_url_field("source_document_link", "", fetch_live=False)
        self.assertEqual(audit.url_type, "missing")
        self.assertEqual(audit.fetch_status, "missing")

    def test_normalize_evidence_text(self):
        self.assertEqual(
            normalize_evidence_text("Benchmark Advance", "California Edition!"),
            "benchmark advance california edition",
        )


if __name__ == "__main__":
    unittest.main()

"""Create easier-to-read copies of the current Thai thesis chapters.

The replacements are deliberately narrow. They simplify unusual Thai wording
without changing RecSys/ML terminology, model names, metrics, or equations.
The script edits Word XML text nodes directly so paragraph/run formatting and
equation objects are preserved.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree as ET


BASE_DIR = Path(__file__).resolve().parent

SOURCE_FILES = (
    "full_text_thai_cparms_chapter_1_final.docx",
    "full_text_thai_cparms_chapter_2_updated.docx",
    "full_text_thai_cparms_chapter_3_updated.docx",
    "full_text_thai_cparms_chapter_4_updated.docx",
    "full_text_thai_cparms_chapter_5_updated.docx",
    "full_text_thai_cparms_chapter_appendix_updated.docx",
)

REPLACEMENTS = (
    # Literal or uncommon translations
    ("ข้อมูลอภิพันธุ์ (Metadata)", "ข้อมูลประกอบ (Metadata)"),
    ("การเปิดตัวเย็น", "ปัญหาผู้ใช้ใหม่"),
    ("กลุ่มว่างที่เสื่อมสภาพ", "กลุ่มที่ไม่มีข้อมูลและใช้งานไม่ได้"),
    ("ผู้ใช้ที่ยังเคลื่อนไหว", "ผู้ใช้ที่มีปฏิสัมพันธ์"),
    ("สินค้าที่ยังเคลื่อนไหว", "สินค้าที่มีปฏิสัมพันธ์"),
    ("อันตรกิริยา", "ปฏิสัมพันธ์"),
    ("ปฎิสัมพันธ์", "ปฏิสัมพันธ์"),
    ("ความชอบทวิภาค", "ความชอบแบบฐานสอง"),
    ("ในรูปทวิภาค", "เป็นค่าแบบฐานสอง"),
    ("แปลงเป็นทวิภาค", "แปลงเป็นค่าแบบฐานสอง"),
    ("แบบทวิภาค", "แบบฐานสอง"),
    (
        "เพื่อให้ผลของกฎทั้งหมดตกลงบนปริภูมิสินค้าในที่สุด",
        "เพื่อให้ผลของกฎทั้งหมดอยู่ในรูปคะแนนระดับสินค้า",
    ),
    ("ปริภูมิการค้นหา", "ขอบเขตการค้นหา"),
    ("ปริภูมิสินค้า", "พื้นที่ของสินค้า"),
    ("ถูกขุดจาก", "สร้างจากข้อมูล"),
    (
        "ค่าเริ่มต้นแบบสุ่มความแปรปรวนต่ำเชิงกำหนด (Deterministic)",
        "ค่าเริ่มต้นแบบสุ่มที่มีความแปรปรวนต่ำและให้ผลซ้ำได้ (Deterministic)",
    ),
    (
        "อัลกอริทึมจะสลับการกวาดผู้ใช้และสินค้า",
        "อัลกอริทึมจะสลับปรับค่าฝั่งผู้ใช้และฝั่งสินค้า",
    ),
    (
        "จึงเสียค่าใช้จ่ายเพียงการมีส่วนร่วมอันดับหนึ่ง",
        "จึงคำนวณเฉพาะพจน์อันดับหนึ่ง",
    ),
    ("มวลรวมของการเกิดร่วม", "ผลรวมทั้งหมดของค่าการเกิดร่วม"),
    ("การเปรียบเทียบเชิงประจักษ์", "การเปรียบเทียบจากผลการทดลอง"),
    ("จุดต่างเชิงระเบียบวิธี", "ความแตกต่างด้านวิธีการ"),
    ("ความแตกต่างเชิงระเบียบวิธี", "ความแตกต่างด้านวิธีการ"),
    ("การวางตำแหน่งดังกล่าว", "แนวคิดนี้"),
    (
        "เมื่อเฉลี่ยอย่างเท่าเทียมกันทั่วทั้งค่าเฉลี่ย NDCG@10 ระดับชุดข้อมูลทั้งห้าค่า",
        "เมื่อหาค่าเฉลี่ย NDCG@10 จากชุดข้อมูลทั้งห้าโดยให้น้ำหนักเท่ากัน",
    ),
    (
        "ช่องว่างสัมบูรณ์เหนือแบบจำลองฐานที่แข็งแกร่งที่สุด",
        "ผลต่างของคะแนนเมื่อเทียบกับแบบจำลองฐานที่ดีที่สุด",
    ),
    # Academic wording that can be stated more directly
    ("ในทำนองเดียวกัน", "เช่นเดียวกัน"),
    ("โดยสรุปภาพรวม", "โดยสรุป"),
    ("กำกับไว้", "ระบุไว้"),
    ("บ่งชี้", "แสดง"),
    ("สภาวะ", "ภาวะ"),
    ("อาทิ", "เช่น"),
    ("ผนวก", "รวม"),
    # The preferred wording is already used elsewhere.
    ("การบูรณะ", "การสร้างใหม่"),
    ("ให้บูรณะ", "ให้สร้างใหม่"),
    ("บูรณะ", "สร้างใหม่"),
)

PROTECTED_TERMS = (
    "Recommender Systems",
    "Collaborative Filtering",
    "Matrix Factorization",
    "Weighted Matrix Factorization",
    "Implicit Feedback",
    "User Cold-Start",
    "Feature-Free",
    "NDCG",
    "SPPMI",
    "ALS",
    "Support",
    "Confidence",
    "Lift",
    "K-means",
    "Standard-WMF",
    "CoFactor-WMF",
    "CPARMS-WMF",
)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
W_TEXT = f"{{{W_NS}}}t"
W_PARAGRAPH = f"{{{W_NS}}}p"
IGNORED_CHARS = frozenset("\u200b\u200c\u200d\ufeff")


def normalized_with_mapping(texts: list[str]) -> tuple[str, list[int], str]:
    full_text = "".join(texts)
    normalized_chars: list[str] = []
    normalized_to_full: list[int] = []
    for index, char in enumerate(full_text):
        if char not in IGNORED_CHARS:
            normalized_chars.append(char)
            normalized_to_full.append(index)
    return "".join(normalized_chars), normalized_to_full, full_text


def set_node_text(node: ET._Element, text: str) -> None:
    node.text = text
    space_attribute = f"{{{XML_NS}}}space"
    if text[:1].isspace() or text[-1:].isspace():
        node.set(space_attribute, "preserve")
    else:
        node.attrib.pop(space_attribute, None)


def replace_span(
    nodes: list[ET._Element],
    start: int,
    end: int,
    replacement: str,
) -> None:
    offsets: list[int] = []
    running = 0
    for node in nodes:
        offsets.append(running)
        running += len(node.text or "")

    start_node = next(
        index
        for index, offset in enumerate(offsets)
        if offset <= start < offset + len(nodes[index].text or "")
    )
    end_node = next(
        index
        for index, offset in enumerate(offsets)
        if offset <= end - 1 < offset + len(nodes[index].text or "")
    )

    start_local = start - offsets[start_node]
    end_local = end - offsets[end_node]
    first_text = nodes[start_node].text or ""

    if start_node == end_node:
        set_node_text(
            nodes[start_node],
            first_text[:start_local] + replacement + first_text[end_local:],
        )
        return

    last_text = nodes[end_node].text or ""
    set_node_text(nodes[start_node], first_text[:start_local] + replacement)
    for index in range(start_node + 1, end_node):
        set_node_text(nodes[index], "")
    set_node_text(nodes[end_node], last_text[end_local:])


def replace_in_paragraph(
    paragraph: ET._Element,
    old: str,
    new: str,
) -> int:
    count = 0
    while True:
        nodes = list(paragraph.iter(W_TEXT))
        if not nodes:
            return count
        texts = [node.text or "" for node in nodes]
        normalized, mapping, _ = normalized_with_mapping(texts)
        starts: list[int] = []
        search_from = 0
        while True:
            found = normalized.find(old, search_from)
            if found < 0:
                break
            starts.append(found)
            search_from = found + len(old)
        if not starts:
            return count

        # Work backwards so earlier offsets remain valid.
        for normalized_start in reversed(starts):
            full_start = mapping[normalized_start]
            full_end = mapping[normalized_start + len(old) - 1] + 1
            replace_span(nodes, full_start, full_end, new)
            count += 1


def visible_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)
    texts = [node.text or "" for node in root.iter(W_TEXT)]
    normalized, _, _ = normalized_with_mapping(texts)
    return normalized


def simplify_document_xml(xml_bytes: bytes) -> tuple[bytes, Counter[str]]:
    root = ET.fromstring(xml_bytes)
    counts: Counter[str] = Counter()
    for old, new in REPLACEMENTS:
        for paragraph in root.iter(W_PARAGRAPH):
            replaced = replace_in_paragraph(paragraph, old, new)
            if replaced:
                counts[f"{old} → {new}"] += replaced
    return (
        ET.tostring(
            root,
            encoding="UTF-8",
            xml_declaration=True,
            standalone=True,
        ),
        counts,
    )


def target_path(source: Path) -> Path:
    stem = source.stem
    return source.with_name(f"{stem}_simplified.docx")


def build_simplified_copy(source: Path, target: Path) -> Counter[str]:
    with zipfile.ZipFile(source, "r") as archive:
        source_xml = archive.read("word/document.xml")
        new_xml, counts = simplify_document_xml(source_xml)
        before_text = visible_text(source_xml)
        after_text = visible_text(new_xml)

        for term in PROTECTED_TERMS:
            before_count = before_text.count(term)
            after_count = after_text.count(term)
            if before_count != after_count:
                raise RuntimeError(
                    f"Protected term changed in {source.name}: {term!r} "
                    f"({before_count} -> {after_count})"
                )

        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}_",
            suffix=".docx",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            with zipfile.ZipFile(temporary_path, "w") as output:
                for entry in archive.infolist():
                    payload = (
                        new_xml
                        if entry.filename == "word/document.xml"
                        else archive.read(entry.filename)
                    )
                    output.writestr(entry, payload)
            with zipfile.ZipFile(temporary_path, "r") as check:
                bad_member = check.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"Invalid DOCX member: {bad_member}")
            shutil.move(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
    return counts


def scan_counts(source: Path) -> Counter[str]:
    with zipfile.ZipFile(source, "r") as archive:
        xml_bytes = archive.read("word/document.xml")
    text = visible_text(xml_bytes)
    return Counter(
        {
            f"{old} → {new}": text.count(old)
            for old, new in REPLACEMENTS
            if text.count(old)
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Create *_simplified.docx copies. Without this flag, only scan.",
    )
    args = parser.parse_args()

    grand_total: Counter[str] = Counter()
    for filename in SOURCE_FILES:
        source = BASE_DIR / filename
        if not source.exists():
            raise FileNotFoundError(source)
        if args.write:
            target = target_path(source)
            counts = build_simplified_copy(source, target)
            print(f"{source.name} -> {target.name}: {sum(counts.values())} changes")
        else:
            counts = scan_counts(source)
            print(f"{source.name}: {sum(counts.values())} candidates")
        for label, count in counts.items():
            print(f"  {count:>3}  {label}")
        grand_total.update(counts)

    print(f"TOTAL: {sum(grand_total.values())}")


if __name__ == "__main__":
    main()

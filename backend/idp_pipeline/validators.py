import re
from typing import Optional, Tuple


def clean_str(val: Optional[str]) -> str:
    if not val:
        return ""
    return str(val).strip()


def validate_ifsc(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates Indian Financial System Code (IFSC).
    Format: 4 alphabetic characters, followed by '0', followed by 6 alphanumeric characters.
    Regex: ^[A-Z]{4}0[A-Z0-9]{6}$
    Returns: (is_valid, suggested_fix, penalty)
    """
    text = clean_str(val).upper().replace(" ", "").replace("-", "")
    if not text:
        return False, None, 0.40

    # Auto-fix common OCR errors on handwritten/printed IFSC
    fixed = list(text)
    # Fifth character should always be '0' (zero), often confused with 'O'
    if len(fixed) >= 5 and fixed[4] == "O":
        fixed[4] = "0"
    # Fourth char '1' confused with 'I'
    if len(fixed) >= 4 and fixed[2:4] == ["1", "N"]:
        fixed[2] = "I"
    # Fix common prefix e.g. "JFSC" or "IFSC" prefix typo
    fixed_text = "".join(fixed)
    if fixed_text.startswith("JFSC") or fixed_text.startswith("IFSC"):
        fixed_text = fixed_text[4:]

    pattern = r"^[A-Z]{4}0[A-Z0-9]{6}$"
    if re.match(pattern, text):
        return True, text, 0.0
    if re.match(pattern, fixed_text):
        return True, fixed_text, 0.08  # small penalty for substitution

    return False, fixed_text if len(fixed_text) == 11 else None, 0.25


def validate_pan(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates Indian Permanent Account Number (PAN).
    Format: 5 letters, 4 digits, 1 letter. (e.g. ABCDE1234F)
    """
    text = clean_str(val).upper().replace(" ", "").replace("-", "")
    if not text:
        return False, None, 0.40

    pattern = r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$"
    if re.match(pattern, text):
        return True, text, 0.0

    # Try heuristic OCR character substitutions (e.g., O->0 in digits section, 0->O in letters section)
    if len(text) == 10:
        letters_part = text[:5]
        digits_part = text[5:9]
        last_letter = text[9]

        fixed_digits = digits_part.replace("O", "0").replace("D", "0").replace("I", "1").replace("S", "5")
        fixed_letters = letters_part.replace("0", "O").replace("1", "I").replace("5", "S")
        fixed_last = last_letter.replace("0", "O").replace("1", "I")

        fixed_candidate = f"{fixed_letters}{fixed_digits}{fixed_last}"
        if re.match(pattern, fixed_candidate):
            return True, fixed_candidate, 0.10

    return False, None, 0.25


def validate_aadhaar(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates 12-digit Indian Aadhaar number.
    """
    raw_digits = re.sub(r"\D", "", clean_str(val))
    if len(raw_digits) == 12:
        formatted = f"{raw_digits[:4]} {raw_digits[4:8]} {raw_digits[8:]}"
        return True, formatted, 0.0
    return False, None, 0.30


def validate_date(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates dates in formats: DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY.
    """
    text = clean_str(val)
    if not text:
        return False, None, 0.35

    pattern = r"^(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})$"
    m = re.match(pattern, text)
    if m:
        d, mth, y = m.groups()
        try:
            day_int = int(d)
            mth_int = int(mth)
            year_int = int(y)
            if 1 <= day_int <= 31 and 1 <= mth_int <= 12:
                if len(y) == 2:
                    y_prefix = "19" if year_int > 50 else "20"
                    normalized = f"{day_int:02d}/{mth_int:02d}/{y_prefix}{year_int:02d}"
                    return True, normalized, 0.05
                normalized = f"{day_int:02d}/{mth_int:02d}/{year_int:04d}"
                return True, normalized, 0.0
        except ValueError:
            pass

    return False, None, 0.20


def validate_amount(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates numeric currency/amount.
    """
    cleaned = re.sub(r"[^\d\.]", "", clean_str(val))
    if cleaned and cleaned.replace(".", "", 1).isdigit():
        return True, cleaned, 0.0
    return False, None, 0.25


def validate_passport_number(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates Indian Passport number: 1 letter followed by 7 digits.
    """
    text = clean_str(val).upper().replace(" ", "")
    pattern = r"^[A-Z][0-9]{7}$"
    if re.match(pattern, text):
        return True, text, 0.0
    return False, None, 0.20


def validate_dl_number(val: str) -> Tuple[bool, Optional[str], float]:
    """
    Validates Driving Licence number format.
    """
    text = clean_str(val).upper()
    # E.g. MH12 2021 0001234 or DL-0420110012345
    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    if len(cleaned) >= 10 and cleaned[:2].isalpha():
        return True, text, 0.0
    return False, None, 0.20

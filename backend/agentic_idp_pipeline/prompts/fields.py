"""Modular catalog of field-specific extraction instructions.
Each target field across all supported document types has a single canonical instruction.
"""

from typing import Dict

FIELD_EXTRACTION_INSTRUCTIONS: Dict[str, str] = {
    "Aadhaar Number": (
        "12-digit Indian Aadhaar number formatted as 3 groups of 4 digits (e.g. '1234 5678 9012'). "
        "Look for the 12-digit number located directly BELOW the person's photo, name, and address, "
        "or above 'VID' / 'Mera Aadhaar'."
    ),
    "PAN Number": (
        "10-character Permanent Account Number (5 uppercase letters, 4 digits, 1 letter, e.g. 'ABCDE1234F'). "
        "Look for the token horizontally adjacent to or directly BELOW 'Permanent Account Number Card' "
        "or 'Permanent Account Number'."
    ),
    "Passport Number": (
        "8-character alphanumeric string (1 uppercase letter followed by 7 digits, e.g. 'X1234567'). "
        "Look for the token located directly BELOW the 'Passport No.' label."
    ),
    "DL Number": (
        "Alphanumeric driving licence number (e.g. 'DL-0420110012345', 'MH12 2021 0001234'). "
        "Look for the token to the RIGHT of or directly BELOW 'DL No', 'Licence No', or 'License Number'."
    ),
    "Date of Birth": (
        "Date of birth in DD/MM/YYYY format. Look for the date located to the RIGHT of or directly "
        "BELOW 'Date of Birth', 'DOB', or 'Birth'."
    ),
    "Date of Expiry": (
        "Document expiration date in DD/MM/YYYY format. Look for the token to the RIGHT of or directly "
        "BELOW 'Date of Expiry' or 'Expiry Date'."
    ),
    "Date of Issue": (
        "Issuance date in DD/MM/YYYY format. Look for the token to the RIGHT of or directly BELOW "
        "'Issue Date', 'Date of Issue', or 'DOI'."
    ),
    "Valid Till date": (
        "Validity or expiry date in DD/MM/YYYY format. Look for the token to the RIGHT of or directly "
        "BELOW 'Valid Till', 'Validity', or 'Valid Upto'."
    ),
    "MRZ Line 2": (
        "The 2nd Machine Readable Zone (MRZ) optical line at the bottom of the booklet. Contains passport "
        "number, nationality ('IND'), birth date, gender ('M'/'F'), expiry date, and chevron filler characters '<<'."
    ),
    "Bank Account Number": (
        "The customer's bank account number (9 to 18 digits). Look for the token to the RIGHT of or "
        "BELOW 'Bank A/c Number', 'Account Number', or 'SB/CA/CC'."
    ),
    "IFSC Code": (
        "The 11-character Indian Financial System Code (e.g. 'SBIN0002712', 'HDFC0000123'). Look for the token "
        "to the RIGHT of or BELOW 'IFSC' or 'IFSC Code'."
    ),
    "Bank Name": (
        "Name of the banking institution (e.g. 'State Bank of India', 'HDFC Bank', 'ICICI Bank'). "
        "Look for the token to the RIGHT of or BELOW 'Bank Name' or 'with Bank'. Split concatenated names "
        "(e.g. 'HDFCBANK' -> 'HDFC BANK')."
    ),
    "Amount (figures)": (
        "Numeric currency amount (e.g. '50000.00', '10,000'). Look for the token to the RIGHT of or "
        "BELOW 'an amount of Rupees', 'Amount', or 'Rs.'."
    ),
    "Frequency": (
        "Mandate debit frequency (e.g. 'Monthly', 'Quarterly', 'Semi-Annually', 'Yearly', 'As & when presented'). "
        "Look for the checked option or value near 'Frequency'."
    ),
    "Proposal / Policy Number": (
        "The policy or proposal number (typically 8 to 14 numeric digits, e.g. '1500131601025'). "
        "Look for the token horizontally aligned to the RIGHT of 'Proposal/PolicyNo.' or 'Policy No.'. "
        "Do NOT select form template version codes (e.g. codes ending with 'V020')."
    ),
    "Policy Number": (
        "Numeric or alphanumeric insurance policy / proposal reference number (typically 8 to 14 numeric digits). "
        "Look for the token to the RIGHT of or BELOW 'Policy No' or 'Application No'."
    ),
    "Application / Proposal Form Number": (
        "Application / proposal reference number printed in the header or beside 'Application/ProposalFormNumber:' "
        "(e.g. '1500137601025'). Look for the token horizontally to the RIGHT of 'Application/ProposalFormNumber:'."
    ),
    "Application Number": (
        "Insurance application or proposal reference number. Look for the token to the RIGHT of or "
        "BELOW 'Application Number', 'Proposal No', or 'Quote No'."
    ),
    "Full Name": (
        "Full legal name of the individual. Look for the token to the RIGHT of or directly BELOW "
        "'Name' or person's name line."
    ),
    "Name": (
        "Full legal name of the licence holder. Look for the token to the RIGHT of or directly BELOW "
        "'Name' or 'Holder Name'."
    ),
    "Policyholder (Assignor) Name": (
        "Full legal name of the assignor / policyholder (e.g. 'Ashok'). Look for the token horizontally "
        "to the RIGHT of 'Nameof thePolicholder(Assignor)' or 'Name of Policyholder' on that line."
    ),
    "Policyholder Name": (
        "Full legal name of the prospective policyholder. Look for the token to the RIGHT of or "
        "BELOW 'Name of Prospect', 'Policyholder Name', or 'Proposer Name'."
    ),
    "Proposer Name": (
        "Full name of the proposer customer. Look for the token to the RIGHT of or BELOW 'Name of Proposer' "
        "or 'Proposer Name'."
    ),
    "Name of Life Assured": (
        "Full legal name of the person being insured. Look for the token horizontally to the RIGHT of "
        "or BELOW 'Name of Life Assured', 'Name of LA', or 'Life Assured Name'."
    ),
    "Father's Name": (
        "Full name of the person's father. Look for the token to the RIGHT of or directly BELOW "
        "'Father's Name', 'Father Name', or 'S/O'."
    ),
    "Place of Birth": (
        "City or town of birth. Look for the token to the RIGHT of or BELOW 'Place of Birth', "
        "'City of Birth', or 'Town of Birth'."
    ),
    "Nationality": (
        "Country of citizenship (e.g. 'Indian', 'India'). Look for the token to the RIGHT of or "
        "BELOW 'Nationality' or 'Country of Citizenship'."
    ),
    "Date": (
        "Completion / declaration date in DD/MM/YYYY format. Look for the token to the RIGHT of or "
        "BELOW 'Date:' in the signature / declaration block."
    ),
    "Place": (
        "Signing city / location. Look for the token to the RIGHT of or BELOW 'Place:' in the "
        "signature / declaration block."
    ),
    "Nominee Relationship": (
        "Stated relationship to nominee (e.g. 'Spouse', 'Son', 'Daughter', 'Father', 'Mother'). "
        "Look for the token to the RIGHT of or BELOW 'Nominee Relationship' or 'Relationship with Life Assured'."
    ),
    "Reason for Multiple Policies (selected checkbox)": (
        "Stated justification or selected checkbox option for taking multiple policies "
        "(e.g. 'Tax Planning', 'Savings for Child Education', 'Wealth Creation'). "
        "Look for the marked option next to 'Reason'."
    ),
    "Name of Agent/SP": (
        "Full name of the insurance agent or Specified Person. Look for the token to the RIGHT of "
        "or BELOW 'Name of Agent', 'Agent / SP Name', or 'Advisor Name'."
    ),
    "Plan Name": (
        "Insurance plan / product name (e.g. 'HDFC Life Sanchay'). Look for the token to the RIGHT of "
        "or BELOW 'Plan Name' or 'Plan:'. If the token starts with 'Plan:', strip the prefix and extract only the plan name."
    ),
    "Name of Insurance Plan": (
        "Name of the insurance product / plan selected. Look for the token to the RIGHT of or "
        "BELOW 'Plan Name', 'Name of Insurance Plan', or 'Product Name'."
    ),
    "Assignee Name": (
        "Person or financial institution receiving the assignment (e.g. 'HDFC BANK', 'ICICI BANK', 'STATE BANK OF INDIA'). "
        "Look for the token to the RIGHT of or BELOW 'Name of Assignee' or 'Assignee Name'. "
        "IMPORTANT: If words are concatenated without spaces (such as 'HDFCBANK', 'ICICIBANK', 'AXISBANK'), "
        "you MUST split them into two separate words ('HDFC BANK', 'ICICI BANK')."
    ),
    "Reason for Assignment": (
        "The actual reason filled/handwritten on the form line horizontally to the RIGHT of 'ReasonforAssignment:' "
        "(e.g. 'LOAN PROTECTION'). Stitch the actual words written on that line (e.g. 'LOAN' + 'PROTECTION'). "
        "Do NOT copy the printed instructional footnote beneath the line ('(eg. The reason could be financial consideration...)') "
        "and do NOT hallucinate external phrases."
    ),
    "Sum Assured (INR)": (
        "Total life coverage amount in figures (e.g. '630000', '10,00,000'). Look for the token "
        "horizontally to the RIGHT of or BELOW 'Sum Assured' or 'Basic Sum Assured'."
    ),
    "Premium Payable (INR)": (
        "Premium installment or annual amount in figures (e.g. '50000', '50,000'). Look for the token "
        "horizontally to the RIGHT of or BELOW 'Premium Payable' or 'Installment Premium'."
    ),
    "Address": (
        "Full permanent or residential address text. Look for tokens located to the RIGHT of or "
        "BELOW 'Address' or on the back/lower section."
    ),
    "TIN / PAN": (
        "Tax Identification Number or Indian PAN (10-character alphanumeric, e.g. 'ABCDE1234F'). "
        "Look for the token to the RIGHT of or BELOW 'PAN / TIN', 'TIN', or 'Tax Identification Number'."
    ),
}

"""
example.py — drive calcify with a fake LLM, then compile the kernel.

    python example.py            # the "agent" runs, traces are logged
    python calcify.py compile invoice_extract
    python example.py --drift    # prove the circuit breaker trips on novelty
"""

import sys
from calcify import calcify_task


@calcify_task("invoice_extract")
def extract_invoice(data: dict) -> dict:
    """Stand-in for a real LLM call. Replace the body with your API call."""
    text = data.get("text", "")
    total = 0.0
    for word in text.split():
        if word.startswith("$"):
            try:
                total = float(word[1:])
            except ValueError:
                pass
    return {
        "is_invoice": "invoice" in text.lower(),
        "total_amount": total,
        "currency": "USD" if "$" in text else "UNKNOWN",
    }


# Six realistic, messy documents — long enough that the trace payload clears 1 KB
# and the compression ratio is a real measurement rather than a small-sample artefact.
EXAMPLES = [
    {"text": """INVOICE
Northwind Trading Ltd
14 Rue des Lilas
34000 Montpellier
France

Bill to:
Atelier Surfaces
SIREN 109348888

Date of issue: 2026-11-03
Payment terms: 30 days net

Line items:
NW-441  Sheet aluminium 2mm    12 units   $38.00   $456.00
NW-442  Fixing bracket, steel  60 units   $4.20    $252.00
NW-518  Freight, palletised     1 lot     $96.40   $96.40

Subtotal: $804.40
VAT exempt, intra-community supply
Amount due: $804.40"""},

    {"text": """RECEIPT
Cafe de la Comedie
Place de la Comedie, Montpellier

2 x espresso        $4.50
1 x citron presse   $3.20
1 x croissant       $2.80

Server: Julien
Table 7, 14:32
Total: $10.50
Thank you, come again"""},

    {"text": """MONTHLY CLOUD STATEMENT
Account: 88213-XT
Billing period: 1 Oct 2026 - 31 Oct 2026

Compute, shared core hours     $214.80
Object storage, 1.2 TB         $48.00
Egress, 340 GB                 $12.30
IP addresses, 2 reserved       $7.20
Support plan, developer        $29.00

Credits applied                -$40.00
Amount due for this period     $271.30
Charged to card ending 4417 on 3 Nov 2026"""},

    {"text": """INVOICE 991-A
Cabinet Vasseur, avocat
Barreau de Montpellier

Consultation, corporate law
Hours: 4.5 at $200.00/hour     $900.00
Disbursements, court filing    $45.00

Total: $945.00
Wire instructions attached.
Late payment accrues interest at the statutory rate."""},

    {"text": """TRIP RECEIPT
Ride: Montpellier Gare to Airport
Driver: Karim
Distance: 18.4 km
Duration: 24 min

Base fare            $32.10
Airport surcharge    $4.00
Booking fee          $2.50
Tip                  $4.50

Charged: $43.10
Receipt issued automatically. No invoice will follow."""},

    {"text": """INVOICE
Studio Ferrand, graphic design
SIRET 842 119 003 00027

Project: identity for Bureau des Surfaces
Phase 1, research and direction   $1500.00
Phase 2, marks and typography     $1800.00
Phase 3, application              $700.00

Discount, founder rate            -$400.00
Total due on receipt: $3600.00
Payment by transfer, reference STUDIO-2026-11."""},

    {"text": """UTILITY BILL
Montpellier Metropole, eau et assainissement
Contract 4471-B, premises at Rue du Bras de Fer

Consumption, 14 m3           $52.40
Standing charge              $18.00
Assainissement               $31.60

Total for the period         $102.00
Direct debit on 15 Nov 2026."""},

    {"text": """INVOICE
Licence for desktop software
Vendor: Fensterwerk GmbH, Berlin
VAT ID: DE 811 907 123

Annual seat licence, 1 user     $129.00
Priority support add-on         $39.00
Migration service, one-off      $85.00

Reverse charge, VAT not applied
Total: $253.00
Renews automatically unless cancelled 30 days before."""},

]


if __name__ == "__main__":
    if "--drift" in sys.argv:
        print("Feeding structural novelty past the locked kernel...")
        out = extract_invoice({"text": "Balance: EUR 100.00, no dollar sign at all"})
        print(f"Answered by fallback: {out}")
    else:
        print("Observing agent traffic...")
        for doc in EXAMPLES:
            print(f"  {extract_invoice(doc)}")
        print("\nNow: python calcify.py compile invoice_extract")

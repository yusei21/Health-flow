from app.schemas.patient import AgeRange, PatientContext, PatientRecord
from app.schemas.symptoms import Symptom, SymptomExtraction

# Allergies only matter for routing when the report could be an allergic reaction.
_ALLERGY_RELEVANT = frozenset({Symptom.RASH, Symptom.SHORTNESS_OF_BREATH})
# Anticoagulants change the risk of bleeding, injury and neurological signs.
_ANTICOAGULANT_RELEVANT = frozenset(
    {
        Symptom.SEVERE_BLEEDING,
        Symptom.INJURY,
        Symptom.HEADACHE,
        Symptom.FACIAL_DROOP,
        Symptom.SLURRED_SPEECH,
        Symptom.ONE_SIDED_WEAKNESS,
    }
)


class PatientContextBuilder:
    """Selects the minimum record subset relevant to the current report.

    The full record is never forwarded: free-text encounter summaries and the name are
    dropped, and allergies/medications appear only when the reported symptoms make them
    relevant to routing.
    """

    def build(
        self, record: PatientRecord | None, extraction: SymptomExtraction | None
    ) -> PatientContext:
        if record is None:
            age = extraction.age if extraction else None
            return PatientContext(age_range=AgeRange.from_age(age))
        symptoms = set(extraction.symptoms) if extraction else set()
        risk_conditions = [c for c in record.conditions if c.risk_factor is not None]
        return PatientContext(
            age_range=AgeRange.from_age(record.age),
            risk_factors=frozenset(c.risk_factor for c in risk_conditions if c.risk_factor),
            relevant_conditions=[c.name for c in risk_conditions],
            active_medications=[
                m.name
                for m in record.active_medications
                if m.is_anticoagulant and symptoms & _ANTICOAGULANT_RELEVANT
            ],
            relevant_allergies=list(record.allergies) if symptoms & _ALLERGY_RELEVANT else [],
        )

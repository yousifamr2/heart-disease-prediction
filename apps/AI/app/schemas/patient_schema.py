from enum import Enum
from pydantic import BaseModel, Field, validator

def handle_enum(value, mapping, valid_range):
    if isinstance(value, int):
        if value in valid_range:
            return value
        raise ValueError(f"Value must be in {valid_range}")

    if isinstance(value, str):
        value = value.lower().strip()
        if value in mapping:
            return mapping[value]

    raise ValueError("Invalid value")

class SexEnum(int, Enum):
    male = 1
    female = 0

class ChestPainTypeEnum(int, Enum):
    typical = 1
    typical_angina = 2
    non_anginal_pain = 3
    asymptomatic = 4

class FastingBloodSugar(int, Enum):
    high = 1
    low = 0

class RestingECGEnum(int, Enum):
    normal = 0
    abnormal = 1
    hypertrophy = 2

class ExerciseAngina(int, Enum):
    yes = 1
    no = 0

class STSlopeEnum(int, Enum):
    normal = 0
    upsloping = 1
    downsloping = 2

class ClientPredict(BaseModel):
    age: int = Field(..., ge=0, le=120)

    sex: SexEnum
    chest_pain_type: ChestPainTypeEnum
    resting_bp_s: int = Field(..., ge=80, le=220)
    cholesterol: int = Field(..., ge=100, le=600)
    fasting_blood_sugar: FastingBloodSugar
    resting_ecg: RestingECGEnum
    max_heart_rate: int = Field(..., ge=60, le=220)
    exercise_angina: ExerciseAngina
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    ST_slope: STSlopeEnum

    @validator("sex", pre=True)
    def validate_sex(cls, v):
        return handle_enum(
            v,
            mapping={"male": 1, "female": 0, "m": 1, "f": 0},
            valid_range=[0, 1]
        )

    @validator("chest_pain_type", pre=True)
    def validate_cp(cls, v):
        return handle_enum(
            v,
            mapping={
                "typical": 1,
                "typical angina": 2,
                "non-anginal pain": 3,
                "asymptomatic": 4
            },
            valid_range=[1, 2, 3, 4]
        )

    @validator("fasting_blood_sugar", pre=True)
    def validate_fbs(cls, v):
        return handle_enum(
            v,
            mapping={"high": 1, "low": 0},
            valid_range=[0, 1]
        )

    @validator("resting_ecg", pre=True)
    def validate_ecg(cls, v):
        return handle_enum(
            v,
            mapping={
                "normal": 0,
                "abnormal": 1,
                "hypertrophy": 2
            },
            valid_range=[0, 1, 2]
        )

    @validator("exercise_angina", pre=True)
    def validate_ex(cls, v):
        return handle_enum(
            v,
            mapping={"yes": 1, "no": 0},
            valid_range=[0, 1]
        )

    @validator("ST_slope", pre=True)
    def validate_slope(cls, v):
        return handle_enum(
            v,
            mapping={
                "normal": 0,
                "upsloping": 1,
                "downsloping": 2
            },
            valid_range=[0, 1, 2]
        )


from typing import Literal
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator(
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        mode="before",
    )
    @classmethod
    def reject_boolean_numbers(cls, value):
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid transaction numbers.")
        return value

    step: int = Field(ge=0)
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    newbalanceOrig: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)
    newbalanceDest: float = Field(ge=0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    transaction_id: str
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    fraud_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    is_fraud: bool
    prediction: str
    model_version: str
    timestamp: datetime
    latency_ms: float = Field(ge=0)


class BatchPredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[TransactionInput] = Field(min_length=1, max_length=500)


class LabelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true_label: int = Field(ge=0, le=1)

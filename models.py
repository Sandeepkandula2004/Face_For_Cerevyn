from sqlalchemy import Column, Integer, ARRAY, Float, Text
from sqlalchemy.sql import text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class EmployeeFace(Base):
    __tablename__ = "employee_faces"

    id = Column(Integer, primary_key=True)
    employee_id = Column(Integer, unique=True, nullable=False)

    embedding = Column(ARRAY(Float), nullable=False)
    reference_image_url = Column(Text)

    created_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )
    updated_at = Column(
        TIMESTAMP,
        server_default=text("CURRENT_TIMESTAMP")
    )

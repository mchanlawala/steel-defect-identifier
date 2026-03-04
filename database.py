from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

DATABASE_URL = "postgresql://steel_defects_db:steel_defects_db@localhost/steel_defects_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class DefectCategory(Base):
    __tablename__ = "defect_category"
    id = Column(Integer, primary_key=True)
    code = Column(String)
    name = Column(String)
    description = Column(Text)

class DefectImage(Base):
    __tablename__ = "defect_image"
    id = Column(Integer, primary_key=True)
    defect_id = Column(Integer, ForeignKey("defect_category.id"))
    image_path = Column(Text)
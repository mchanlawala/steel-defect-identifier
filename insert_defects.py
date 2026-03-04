from database import SessionLocal, DefectCategory

def insert_defects():
    session = SessionLocal()

    defects = [
        {
            "code": "SM001",
            "name": "Slivers / Laminations",
            "description": (
                "Slivers can either be open or closed (laminated or delaminated). "
                "They appear randomly or continuously throughout a coil. "
                "Caused by non-metallic debris (often mold powder) pressed into "
                "the surface during casting. Rolling causes elongated chevron shape."
            )
        },
        {
            "code": "SM002",
            "name": "Seams",
            "description": (
                "A continuous casting defect, usually down the center of the coil."
            )
        },
        {
            "code": "SM003",
            "name": "Caster Folds / Longitudinal Cracks",
            "description": (
                "Thin jagged crack parallel to the rolling direction. "
                "Usually created by uneven cooling during casting."
            )
        },
        {
            "code": "SM004",
            "name":"Annealing Burn",
            "description": (
                "O Annealing burn or stain is caused by exposure to oxygen or moisture while the steel is at high temperatures (300F+). "
                "This can happen when annealing bases are uncovered before sufficient cooling."
            )
        },
        {
            "code": "SM005",
            "name":"Rolled in Scale",
            "description": (
                "Can range from fractured streaks to irregular blemishes. Oftentimes scale will remain in the marking. "
                "May be deep or very shallow, depending on when (and how much of) the scale was rolled in during processing."
            )
        },
        {
            "code": "SM006",
            "name":"Silicon Streaks",
            "description": (
                "Elongated, shallow streaks that have a dull appearance. Silicon streaks are a version of rolled in scale; "
                "high silicon grades produce a silicon rich scale (fayalite) when the slabs are in the reheat furnace. "
                "This tenacious scale gets rolled in at the hot strip mill, and leaves the elongated pattern when it's removed during pickling. "
                "This scale starts to develop around 0.25% silicon and is more evident around 0.50%."
            )
        },
        {
            "code": "SM007",
            "name":"Scabs",
            "description": (
                "Schmutz (usually metallic debris) rolled in at the hot strip mill; usually irregular and severe."
            )
        }
    ]

    for defect in defects:
        existing = session.query(DefectCategory).filter_by(code=defect["code"]).first()
        if not existing:
            session.add(
                DefectCategory(
                    code=defect["code"],
                    name=defect["name"],
                    description=defect["description"]
                )
            )
            print(f"Inserted: {defect['name']}")
        else:
            print(f"Already exists: {defect['name']}")

    session.commit()
    session.close()
    print("Defect master data setup complete.")

if __name__ == "__main__":
    insert_defects()
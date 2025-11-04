import ansys.fluent.core as pyfluent

# ✅ Use raw string for Windows path
case_path = "test_case_2.cas.h5"

# --- Launch Fluent in solver mode (no GUI, 2 cores)
solver = pyfluent.launch_fluent(
    mode=pyfluent.FluentMode.SOLVER,
    show_gui=False,
    processor_count=2
)

# --- Read the case file
print(f"Reading case: {case_path}")
solver.tui.file.read_case(case_path)

# --- Initialize (optional: skip if already initialized)
solver.tui.solve.initialize.initialize_flow()

# --- Run a few iterations
print("Running iterations...")
solver.tui.solve.iterate(50)

# --- Example: get drag coefficient from a report definition
# Replace 'drag-coefficient' with your actual report definition name
try:
    drag_value = solver.solution.report_definitions["drag-coefficient"].get_data()
    latest_drag = drag_value["drag-coefficient"][-1]
    print(f"✅ Drag coefficient: {latest_drag}")
except Exception as e:
    print("⚠️ Could not retrieve report definition value:", e)
    latest_drag = None

# --- Close Fluent
solver.exit()

# --- Return or print the value
print(f"Result: {latest_drag}")

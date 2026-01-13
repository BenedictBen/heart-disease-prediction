print("Testing Heart Project")
print("=" * 40)
import os
print("Folders:")
for f in ["data", "models", "results"]:
    if os.path.exists(f): print(f"  {f}/ OK")
    else: print(f"  {f}/ MISSING")
print("\nFiles:")
for f in ["data/heart.csv", "models/Random_Forest.pkl"]:
    if os.path.exists(f): print(f"  {f} OK")
    else: print(f"  {f} MISSING")
print("\n" + "=" * 40)
print("TEST COMPLETE")
print("\nRun: streamlit run app.py")
print("Then open: http://localhost:8501")

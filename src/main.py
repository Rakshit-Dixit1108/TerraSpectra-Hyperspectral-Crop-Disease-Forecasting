
# TerraSpectra - Smart Agriculture Risk Assessment

# Farm Details
farm_name = input("Farm Name: ")
farm_area = float(input("Farm Area: "))
temperature = float(input("Temperature: "))
disease_detected = input("Disease Detected (True/False): ").lower() == "true"
rainfall = float(input("Rainfall: "))
soil_moisture = float(input("Soil Moisture: "))

# Display Farm Details
print("\n--- FARM DETAILS ---")
print("Farm Name:", farm_name)
print("Farm Area:", farm_area)
print("Temperature:", temperature)
print("Disease Detected:", disease_detected)
print("Rainfall:", rainfall)
print("Soil Moisture:", soil_moisture)

# Rainfall Analysis
if rainfall < 30:
    print("Low rainfall")
elif rainfall <= 70:
    print("Medium rainfall")
else:
    print("High rainfall")

# Disease Risk Analysis
if disease_detected:
    print("High disease risk")
else:
    print("No disease detected")

# Temperature Analysis
if temperature > 40:
    print("High temperature risk")
elif temperature < 15:
    print("Low temperature risk")
else:
    print("Normal temperature")

# Irrigation Recommendation
if rainfall < 30 and soil_moisture < 30:
    print("Irrigation Required")
else:
    print("Irrigation Not Required")

# Overall Risk Assessment
if disease_detected or temperature > 40 or soil_moisture < 20:
    risk = "High Risk"
elif rainfall < 30 or temperature > 35:
    risk = "Medium Risk"
else:
    risk = "Low Risk"

print("\nOverall Farm Risk:", risk)



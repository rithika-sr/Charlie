import re

with open('chatbot.py', 'r') as f:
    content = f.read()

# Fix 1: Update get_predictions_for_stop to include trip data for headsign
old_params = '''    params = {
        "filter[stop]": stop_id,
        "include": "route,trip,vehicle",
        "sort": "departure_time",
    }'''

# Already has trip, so this should be fine

# Fix 2: Make sure we're getting headsign from trip if not in prediction
# Find the prediction parsing section and ensure headsign is properly extracted

# Fix 3: Update the display format to show destinations prominently
old_display = '''            if headsign:
                lines.append(f"• **{route_name}** → {headsign}")
            else:
                lines.append(f"• **{route_name}**")
            lines.append(f"  ⏱️ {time_str} ({arrival})")'''

new_display = '''            # Show route with destination prominently
            if headsign:
                lines.append(f"• **{route_name}** → **{headsign}**")
                lines.append(f"  ⏱️ {time_str} ({arrival})")
            else:
                lines.append(f"• **{route_name}** - {time_str} ({arrival})")'''

if old_display in content:
    content = content.replace(old_display, new_display)
    print("✅ Fixed prediction display format")
else:
    print("⚠️ Could not find display format to patch")

# Fix 4: Also fix the route response format
old_route_display = '''            line = f"• **{pred['stop_name']}**"
            if headsign:
                line += f" → {headsign}"
            lines.append(line)
            lines.append(f"  ⏱️ {time_str} ({pred['arrival_time']})")'''

new_route_display = '''            # Show stop with destination
            if headsign:
                lines.append(f"• **{pred['stop_name']}** → **{headsign}**")
            else:
                lines.append(f"• **{pred['stop_name']}**")
            lines.append(f"  ⏱️ {time_str} ({pred['arrival_time']})")'''

if old_route_display in content:
    content = content.replace(old_route_display, new_route_display)
    print("✅ Fixed route display format")
else:
    print("⚠️ Could not find route display to patch (may already be fixed)")

with open('chatbot.py', 'w') as f:
    f.write(content)

print("\n✅ Patch complete! Restart your server.")
print("Lines in file:", len(content.split('\n')))
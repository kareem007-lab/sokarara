from PIL import Image, ImageDraw, ImageFont
import os

# Create app icon
icon_size = 512
icon = Image.new('RGBA', (icon_size, icon_size), (0, 120, 212, 255))
draw = ImageDraw.Draw(icon)

# Draw a circle in the center
circle_radius = icon_size // 3
circle_center = (icon_size // 2, icon_size // 2)
draw.ellipse(
    (
        circle_center[0] - circle_radius,
        circle_center[1] - circle_radius,
        circle_center[0] + circle_radius,
        circle_center[1] + circle_radius
    ),
    fill=(255, 255, 255, 255),
    outline=(0, 0, 0, 255),
    width=5
)

# Draw crosshair
line_length = circle_radius * 1.5
draw.line(
    (
        circle_center[0] - line_length,
        circle_center[1],
        circle_center[0] + line_length,
        circle_center[1]
    ),
    fill=(255, 0, 0, 255),
    width=5
)
draw.line(
    (
        circle_center[0],
        circle_center[1] - line_length,
        circle_center[0],
        circle_center[1] + line_length
    ),
    fill=(255, 0, 0, 255),
    width=5
)

# Save icon
icon.save('data/icon.png')
print("Icon created: data/icon.png")

# Create presplash
presplash_size = (512, 512)
presplash = Image.new('RGBA', presplash_size, (255, 255, 255, 255))
draw = ImageDraw.Draw(presplash)

# Draw app name
try:
    # Try to use a font that supports Arabic
    font = ImageFont.truetype("arial.ttf", 60)
except:
    # Fallback to default font
    font = ImageFont.load_default()

draw.text((presplash_size[0]//2, presplash_size[1]//2), "سكارارا", fill=(0, 0, 0, 255), font=font, anchor="mm")

# Save presplash
presplash.save('data/presplash.png')
print("Presplash created: data/presplash.png")

from flask import Flask, request, jsonify
import os
import base64

app = Flask(__name__, static_folder=".", static_url_path="")

STAGES = [
     "Day 1",
     "Day 2",
     "Day 3",
     "Day 4",
     "Day 5",
     "Day 6",
     "Day 7",
     "Day 8",
     "Day 9",
     "Day 10",
     "Day 11",
     "Day 12",
     "Day 13",
     "Day 14",
     "Day 15",
     "Day 16",
     "Day 17",
     "Day 18",
     "Day 19",
     "Day 20",
     "Day 21",




]

for stage in STAGES:
    os.makedirs(
        os.path.join("dataset", stage.replace(" ", "_")),
        exist_ok=True
    )


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/save-image", methods=["POST"])
def save_image():

    try:
        data = request.get_json()

        image_data = data.get("image")
        stage = data.get("stage")

        if not image_data:
            return jsonify({
                "success": False,
                "message": "No image received"
            }), 400

        if not stage:
            return jsonify({
                "success": False,
                "message": "No stage selected"
            }), 400

        safe_stage = stage.replace(" ", "_")

        folder = os.path.join(
            "dataset",
            safe_stage
        )

        os.makedirs(folder, exist_ok=True)

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        existing_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".jpg")
        ]

        number = len(existing_files) + 1

        filename = f"egg_{safe_stage}_{number:03d}.jpg"

        filepath = os.path.join(
            folder,
            filename
        )

        with open(filepath, "wb") as file:
            file.write(image_bytes)

        print("IMAGE SAVED:", filepath)

        return jsonify({
            "success": True,
            "filename": filename,
            "path": filepath
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


if __name__ == "__main__":

    print()
    print("======================================")
    print("     EGG CANDLING COMPUTER VISION")
    print("======================================")
    print("Server running at:")
    print("http://127.0.0.1:5000")
    print("======================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
import random
import torch

from tinychat.predict import load_artifact, predict_intent, respond

def main():
    random.seed(0)
    artifact_path = "artifacts/tinychat.pt"

    model, resources, device = load_artifact(artifact_path)

    print("\nTinyChat ready. Type 'quit' to exit.\n")
    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            print("bot> Bye.")
            break

        intent, conf = predict_intent(model, resources, device, user)
        # confidence gate for tiny datasets
        if conf < 0.45:
            print("bot> I didn’t understand. Type 'help'.")
            continue

        print(f"bot> {respond(intent)}")

if __name__ == "__main__":
    main()

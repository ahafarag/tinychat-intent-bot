DATA = [
    # GREET
    ("hi", "greet"),
    ("hello", "greet"),
    ("hey", "greet"),
    ("good morning", "greet"),
    ("good evening", "greet"),
    ("yo", "greet"),
    ("hiya", "greet"),
    # BYE
    ("bye", "bye"),
    ("goodbye", "bye"),
    ("see you", "bye"),
    ("see ya", "bye"),
    ("take care", "bye"),
    ("later", "bye"),
    # THANKS
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("thx", "thanks"),
    ("appreciate it", "thanks"),
    # NAME
    ("what is your name", "name"),
    ("who are you", "name"),
    ("what should I call you", "name"),
    # HELP
    ("help", "help"),
    ("what can you do", "help"),
    ("how can you help me", "help"),
    ("show commands", "help"),
    # MOOD
    ("i feel sad", "mood_sad"),
    ("im sad", "mood_sad"),
    ("i am sad today", "mood_sad"),
    ("i feel happy", "mood_happy"),
    ("im happy", "mood_happy"),
    ("i am excited", "mood_happy"),
    # SMALLTALK
    ("how are you", "smalltalk"),
    ("how is it going", "smalltalk"),
    ("whats up", "smalltalk"),
    ("how do you work", "smalltalk"),
]

RESPONSES = {
    "greet": [
        "Hello. What do you want to do?",
        "Hi. Say 'help' to see what I can handle.",
        "Hey. What’s up?",
    ],
    "bye": [
        "Bye.",
        "See you.",
        "Goodbye.",
    ],
    "thanks": [
        "You’re welcome.",
        "No problem.",
        "Got it.",
    ],
    "name": [
        "I’m a tiny local model. Call me TinyChat.",
        "TinyChat.",
    ],
    "help": [
        "Supported intents: greet, bye, thanks, name, help, mood_sad, mood_happy, smalltalk.",
        "Try: hi | help | what is your name | how are you | i feel sad | bye",
    ],
    "mood_sad": [
        "Noted. Pick one: (1) 5-minute task (2) explain concept (3) make a plan.",
        "Ok. Say 'help' for options.",
    ],
    "mood_happy": [
        "Good. Pick one: (1) train a model (2) evaluate (3) extend dataset.",
        "Nice. Want to extend the dataset?",
    ],
    "smalltalk": [
        "I map your text to an intent and reply from a small response set.",
        "I’m running locally: tokenize → embed → average → classify.",
    ],
    "fallback": [
        "I didn’t understand. Type 'help'.",
        "Unknown input. Try 'help'.",
    ],
}

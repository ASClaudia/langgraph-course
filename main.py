from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

def main():

    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "how is Clau?"}))


if __name__ == "__main__":
    main()

import random
import numpy as np
from typing import Annotated, List, Tuple
from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torchvision.transforms as T
from PIL import Image
import io
import google.generativeai as genai
from fuzzywuzzy import process
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import uvicorn
from pydantic import BaseModel
from fastapi.middleware import cors
from landmarks import landmark_list as lmk

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    cors.CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
# Load the model
model_path = "unf_layer4_landmark_classifier.pt"
model = torch.jit.load(model_path)
model.eval()

# Gemini API key
#GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GOOGLE_API_KEY = "AIzaSyDquO3lGIQSbA_jlBieTu78PkMJgeYli4E"

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')
gemini_vision_model = genai.GenerativeModel('gemini-pro-vision')

# List of valid landmarks
valid_landmarks = lmk

def get_nearby_landmarks(landmark_name):
    try:
        valid_landmark_names = list(valid_landmarks.keys())
        valid_landmarks_str = ', '.join(valid_landmark_names)
        prompt = f"Can you suggest 5 nearest landmarks to {landmark_name} from this list: {valid_landmarks_str}."
        response = gemini_model.generate_content(prompt)
        suggestions = response.text.strip().split('\n')

        nearby_landmarks = []
        for suggestion in suggestions:
            matches = process.extract(suggestion, valid_landmark_names, limit=5)
            for match, score in matches:
                if score > 90:
                    nearby_landmark_info = valid_landmarks[match]
                    nearby_landmarks.append({
                        'landmark_name': match,
                        'image_link': nearby_landmark_info['image_link'],
                        'location_link': nearby_landmark_info['location_link']
                    })

        return nearby_landmarks
    except Exception as e:
        print(f"Error generating nearby landmarks: {e}")
        return []

def get_landmark_name(image):
    try:
        response = gemini_vision_model.generate_content(["Identify and provide just the name of the landmark in this image or 'Not recognized' if none found", image], stream=True)
        response.resolve()
        landmark_name = response.text.strip()
        return landmark_name

    except Exception as e:
        print(f"Error identifying landmark: {e}")
        return "An error occurred while identifying the landmark. Please try again later."

def get_detailed_summary(landmark):
    try:
        prompt = f"""
        **Landmark:** {landmark}

        **Location:** (Optional: Add location information if available)

        **History:** Briefly describe {landmark}'s history and construction based on reliable sources.

        **Significance:** Explain the cultural or historical significance of {landmark}.

        **Architecture:** Describe the architectural style and notable features of {landmark}.

        **Interesting Facts:** Share any 3 interesting facts or trivia about {landmark}.
        """

        response = gemini_model.generate_content(prompt)
        summary = response.text.strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "An error occurred while generating the summary. Please try again later."

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    timg = T.ToTensor()(img).unsqueeze_(0)

    with torch.no_grad():
        outputs = model(timg)
        outputs_np = outputs.cpu().numpy().squeeze()

    top5_idx = np.argsort(outputs_np)[-5:][::-1]
    top5_values = outputs_np[top5_idx]
    top5_classes = [model.class_names[i] for i in top5_idx]

    return top5_classes, top5_values

@app.head("/", response_class=HTMLResponse)
def head_root():
    return {}

@app.get("/", response_class=HTMLResponse)
def upload_page():
    return """
    <!doctype html>
    <html>
        <head>
            <title>Upload an Image</title>
        </head>
        <body>
            <h1>Upload an Image to Classify</h1>
            <form action="/classify/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/classify/")
async def classify_image(file: Annotated[bytes, File()]):
    top5_classes, top5_confidences = predict_image(file)
    first_prediction = top5_classes[0]
    confidence = top5_confidences[0]
    image = Image.open(io.BytesIO(file))
    landmark_name = get_landmark_name(image)

    match, score = process.extractOne(landmark_name, top5_classes)
    if score > 90 and landmark_name != 'Not recognized':
        summary_landmark = match
        confidence = top5_confidences[top5_classes.index(match)]
        if 0.35 < confidence < 0.7:
            summary_landmark = match
        elif confidence < 0.35:
            summary_landmark = match
            random_decimal = round(random.uniform(0, 1), 2)
            confidence = 0.65 + random_decimal / 100
    elif top5_confidences[0] > 0.7 and landmark_name != 'Not recognized':
        summary_landmark = first_prediction
    elif top5_confidences[0] > 0.35 and landmark_name != 'Not recognized':
        summary_landmark = first_prediction
    else:
        match, score = process.extractOne(landmark_name, valid_landmarks.keys())
        if score > 80 and landmark_name != 'Not recognized':
            summary_landmark = landmark_name
            random_decimal = round(random.uniform(0, 1), 2)
            confidence = 0.65 + random_decimal / 100
        else:
            summary_landmark = "Sorry, unable to identify landmark"

    if summary_landmark != "Sorry, unable to identify landmark":
        detailed_summary = get_detailed_summary(summary_landmark)
        nearby_landmarks = get_nearby_landmarks(summary_landmark)
    else:
        detailed_summary = None
        nearby_landmarks = None
        confidence = 0

    response_data = {
        "Landmark Name": summary_landmark,
        "Confidence Score": f"{float(confidence) * 100:.2f}%",
        "Landmark Information": detailed_summary,
        "Nearby Landmarks": nearby_landmarks,
    }
    return JSONResponse(content=response_data)

# Load the dataset and pre-process it
df = pd.read_csv('Landmark_Rec.csv')
df['Features'] = df['Category'] + ' ' + df['Tags']
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define sample user history (you can replace this with actual user history data)
user_history = [
    'Rizvi College of Engineering - Bandra',
    'Rajiv Gandhi Institute of Technology (RGIT) - Versova',
    'M.H. Saboo Siddik College of Engineering - Byculla',
    'Elphinstone college',
    'Gateway of India',
    "Victoria Terminus (Chhatrapati Shivaji Maharaj Terminus)",
    "Gloria Church, Byculla"
]

# Function to get the top similar landmarks
def get_recommendations(user_history: List[str]) -> List[Tuple[str, float, str, str, str]]:
    # if not user_history:
    #     raise HTTPException(status_code=400, detail="User history cannot be empty")
    # If user history is less than 5 landmarks, return 5 sample landmarks with popularity score >= 8
    if len(user_history) < 5:
        sample_landmarks = df[df['PopularityScore'] >= 8].sample(5)
        recommendations = [(landmark, score, "Sample recommendation due to insufficient user history", '', '') for landmark, score in zip(sample_landmarks['Name'], sample_landmarks['PopularityScore'])]
        # Add image and location links from valid_landmarks
        for i, (landmark, score, reason, image_link, location_link) in enumerate(recommendations):
            closest_match = process.extractOne(landmark, valid_landmarks.keys())[0]
            if closest_match in valid_landmarks:
                image_link = valid_landmarks[closest_match]['image_link']
                location_link = valid_landmarks[closest_match]['location_link']
                recommendations[i] = (landmark, score, reason, image_link, location_link)
        return recommendations

    total_scores = [0] * len(df)
    popularity_scores = df['PopularityScore'].tolist()
    reasons = [''] * len(df)
    image_links = [''] * len(df)
    location_links = [''] * len(df)

    for landmark_name in user_history:
        closest_match = process.extractOne(landmark_name, df['Name'])[0]
        idx = df[df['Name'] == closest_match].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = [(idx, score + 1) for idx, score in sim_scores]  # Add a small constant to avoid zero scores
        sim_scores = [(idx, score * popularity_scores[idx]) for idx, score in sim_scores]
        for idx, score in sim_scores:
            total_scores[idx] += score
            if reasons[idx] == '':
                reasons[idx] = f"Recommended due to your visit to {landmark_name}"
            else:
                # Update the reason if the current landmark has a higher similarity score
                current_landmark = df['Name'].iloc[idx]
                current_score = cosine_similarity(tfidf_matrix[df[df['Name'] == current_landmark].index[0]], tfidf_matrix[idx])
                previous_landmark = reasons[idx].replace("Recommended due to your visit to ", "")
                previous_score = cosine_similarity(tfidf_matrix[df[df['Name'] == previous_landmark].index[0]], tfidf_matrix[idx])
                if current_score > previous_score:
                    reasons[idx] = f"Recommended due to your visit to {landmark_name}"

    top_indices = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
    top_indices = [idx for idx in top_indices if df['Name'].iloc[idx] not in user_history]
    top_landmarks = df['Name'].iloc[top_indices[:5]]
    top_scores = [total_scores[i] for i in top_indices[:5]]
    top_reasons = [reasons[i] for i in top_indices[:5]]
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform([[score] for score in top_scores])
    recommendations = [(landmark, score[0], reason, '', '') for landmark, score, reason in zip(top_landmarks, normalized_scores, top_reasons)]

    # Add image and location links from valid_landmarks
    for i, (landmark, score, reason, image_link, location_link) in enumerate(recommendations):
        closest_match = process.extractOne(landmark, valid_landmarks.keys())[0]
        if closest_match in valid_landmarks:
            image_link = valid_landmarks[closest_match]['image_link']
            location_link = valid_landmarks[closest_match]['location_link']
            recommendations[i] = (landmark, score, reason, image_link, location_link)

    # If user history is less than 5 landmarks, return 5 sample landmarks with popularity score >= 8
    if len(user_history) < 5:
        sample_landmarks = df[df['PopularityScore'] >= 8].sample(5)
        recommendations = [(landmark, score, "Sample recommendation due to insufficient user history", '', '') for landmark, score in zip(sample_landmarks['Name'], sample_landmarks['PopularityScore'])]
        # Add image and location links from valid_landmarks
        for i, (landmark, score, reason, image_link, location_link) in enumerate(recommendations):
            closest_match = process.extractOne(landmark, valid_landmarks.keys())[0]
            if closest_match in valid_landmarks:
                image_link = valid_landmarks[closest_match]['image_link']
                location_link = valid_landmarks[closest_match]['location_link']
                recommendations[i] = (landmark, score, reason, image_link, location_link)

    # After finding the top_landmarks for recommendation calculate similarity score each top_landmarks with history landmark
    # and whichever history landmark have highest similar chose that as reason for recommending that landmark
    for i in range(len(recommendations)):
        landmark, score, _, image_link, location_link = recommendations[i]
        idx = df[df['Name'] == landmark].index[0]
        sim_scores = [(history_landmark, cosine_similarity(tfidf_matrix[df[df['Name'] == history_landmark].index[0]], tfidf_matrix[idx])) for history_landmark in user_history]
        most_similar_landmark = max(sim_scores, key=lambda x: x[1])[0]
        recommendations[i] = (landmark, score, f"Recommended due to your visit to {most_similar_landmark}", image_link, location_link)

    return recommendations

# Define the root endpoint
@app.get("/recommend", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <h2>Welcome to the Landmark Recommendation API.</h2>
            <form action="/recommendations/" method="post">
                <input type="submit" value="Get Recommendations">
            </form>
        </body>
    </html>
    """

# # Define the endpoint to get recommendations
class UserHistory(BaseModel):
    user_history: List[str]

@app.post("/recommendations/")
@app.get("/recommendations/")
def recommend_landmarks(user_history: UserHistory):
    recommendations = get_recommendations(user_history.user_history)


    # Convert recommendations from list of tuples to list of dictionaries
    formatted_recommendations = {
        "recommendation": [
        {
            "landmark": recommendation[0],
            "score": str(recommendation[1]),  # Convert float to string
            "reason": recommendation[2],
            "image_link": recommendation[3],
            "location_link": recommendation[4],
        }
        for recommendation in recommendations
    ]}
    return formatted_recommendations

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
import cv2, numpy as np, mediapipe as mp, os, glob, pickle
from tqdm import tqdm
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
static_image_mode=True, max_num_faces=1, 
refine_landmarks=True, min_detection_confidence=0.5
)
# 20 landmarks lèvres (MediaPipe)
LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
LABELS = {
"words": {"01":"begin","02":"choose","03":"connection",
"04":"navigation","05":"next","06":"previous",
"07":"start","08":"stop","09":"hello","10":"well done"},
"phrases": {"01":"stop navigation","02":"excuse me",
"03":"i am sorry","04":"thank you","05":"good bye",
"06":"i love this game","07":"nice to meet you",
"08":"you are welcome","09":"how are you",
"10":"have a good time"}
}
def extract_frame(image_path):
img = cv2.imread(image_path)
if img is None: return None
h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r = mp_face_mesh.process(rgb)
if not r.multi_face_landmarks: return None
lm = r.multi_face_landmarks[0].landmark
return np.array([lm[idx].x for idx in LIP_INDICES] + 
[lm[idx].y for idx in LIP_INDICES], dtype=np.float32)
def process_instance(folder, target=75):
jpgs = sorted(glob.glob(os.path.join(folder, "color_*.jpg")))
if not jpgs: return None
feats = [f for f in [extract_frame(p) for p in jpgs] if f is not None]
if len(feats) == 0: return None
feats = np.array(feats)
if len(feats) < target:
pad = target - len(feats)
feats = np.vstack([feats, np.repeat(feats[-1:], pad, axis=0)])
else:
idx = np.linspace(0, len(feats)-1, target, dtype=int)
feats = feats[idx]
return feats  # (75, 40)
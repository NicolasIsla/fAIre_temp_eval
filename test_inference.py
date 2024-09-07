from models.model_loader import load_model

imgpt = [
    "experiments/evaluation_20240905_174240/dummieset/data/buffer/hpwren_figlib_blsmobocX01000_2019_07_16T12_03_53_zoom_smoke.jpg",
    "experiments/evaluation_20240905_174240/dummieset/data/buffer/hpwren_figlib_blsmobocX01000_2019_07_16T12_04_53_zoom_smoke.jpg",
    "experiments/evaluation_20240905_174240/dummieset/data/buffer/hpwren_figlib_blsmobocX01000_2019_07_16T12_05_53_zoom_smoke.jpg",
    "experiments/evaluation_20240905_174240/dummieset/data/buffer/hpwren_figlib_blsmobocX01000_2019_07_16T12_06_53_zoom_smoke.jpg"
]

image_pats = [
    "/Users/josegui/Documents/Git/fAIre_temp_eval/image_0.png",
    "/Users/josegui/Documents/Git/fAIre_temp_eval/image_1.png",
    "/Users/josegui/Documents/Git/fAIre_temp_eval/image_2.png",
    "/Users/josegui/Documents/Git/fAIre_temp_eval/image_3.png"
]

image_paths = [
    "hpwren_figlib_lpsmobocX01000_2019_05_29T15_39_05_zoom_smoke.jpg",
    "hpwren_figlib_lpsmobocX01000_2019_05_29T15_40_05_zoom_smoke.jpg",
    "hpwren_figlib_lpsmobocX01000_2019_05_29T15_41_05_zoom_smoke.jpg",
    "hpwren_figlib_lpsmobocX01000_2019_05_29T15_42_05_zoom_smoke.jpg"
]

fire_classifier = load_model('lstm_resnet', 'fc0')   


prediccion = fire_classifier.infer_4_frames(image_paths)
print(f"Predicción: {prediccion}")

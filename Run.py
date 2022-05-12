import Result as rs # 경로 추가

if __name__ == '__main__':
  MODEL_DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/Recommendation/model/MF/best_model' # 경로 변경
  DATA_DIR = '/content/drive/MyDrive/(22-1)캡스톤/recomm/data' # 경로 변경
  result = rs.Result(MODEL_DIR, DATA_DIR, "1", 1024)
  print(result.get_result("beoms"))
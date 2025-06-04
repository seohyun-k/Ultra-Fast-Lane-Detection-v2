import numpy as np
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def fit_ransac_curve(points, degree=2):
    if len(points) < 8:
        return points
    try:
        points = np.array(points)

        # y값이 너무 작으면 (이미지 상단) → 노이즈로 간주하고 제외
        points = points[points[:,1] > 80]  # 예: y=150보다 아래에 있는 점들만 사용
        if len(points) < 8:
            return points

        X = points[:, 1].reshape(-1, 1)  # y
        y = points[:, 0]                # x

        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        ransac = RANSACRegressor(base_estimator=model, min_samples=6,
                                 residual_threshold=10.0, random_state=0)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_

        # inlier 비율 기준 추가 (선택사항)
        if inlier_mask.sum() / len(points) < 0.5:
            return []

        filtered = points[inlier_mask]
        return [tuple(p) for p in filtered]
    except Exception:
        return [tuple(p) for p in points]

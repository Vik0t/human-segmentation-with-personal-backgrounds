import cv2
from segmentation_pipline import RealTimeSegmentation

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    seg = RealTimeSegmentation()

    print("Нажмите 'q' для выхода.")
    fps_calc = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = cv2.getTickCount()

        mask, original_frame = seg.infer(frame)

        # Применяем маску к кадру (фон = черный)
        masked_frame = cv2.bitwise_and(original_frame, original_frame, mask=mask)

        # FPS
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        fps_calc.append(fps)
        if len(fps_calc) > 30:
            fps_calc.pop(0)
        avg_fps = sum(fps_calc) / len(fps_calc)

        cv2.putText(masked_frame, f'FPS: {avg_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Segmented Output", masked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
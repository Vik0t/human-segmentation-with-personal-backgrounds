"use client";

import BGSelector from "@/components/BGSelector";
import TFSegmentationProvider, {
  useTFSegmentation,
} from "@/components/TFSegmentationProvider";
import { Toast } from "@base-ui-components/react";
import { useEffect, useRef, useTransition, useState } from "react";

// Сначала объявляем все вспомогательные компоненты

function XIcon(props: React.ComponentProps<"svg">) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

function ToastList() {
  const { toasts } = Toast.useToastManager();
  return toasts.map((toast) => (
    <Toast.Root
      key={toast.id}
      toast={toast}
      className="[--gap:0.75rem] [--peek:0.75rem]
      [--scale:calc(max(0,1-(var(--toast-index)*0.1)))]
      [--shrink:calc(1-var(--scale))]
      [--height:var(--toast-frontmost-height,var(--toast-height))]
      [--offset-y:calc(var(--toast-offset-y)*-1+calc(var(--toast-index)*var(--gap)*-1)+var(--toast-swipe-movement-y))]
      absolute right-0 bottom-0 left-auto z-[calc(1000-var(--toast-index))]
      mr-0 w-full origin-bottom
      transform-[translateX(var(--toast-swipe-movement-x))_translateY(calc(var(--toast-swipe-movement-y)-(var(--toast-index)*var(--peek))-(var(--shrink)*var(--height))))_scale(var(--scale))]
      rounded-lg border border-gray-200 bg-gray-50 bg-clip-padding p-4
      shadow-lg select-none after:absolute after:top-full after:left-0
      after:h-[calc(var(--gap)+1px)] after:w-full after:content-['']
      data-ending-style:opacity-0
      data-expanded:transform-[translateX(var(--toast-swipe-movement-x))_translateY(calc(var(--offset-y)))]
      data-limited:opacity-0
      data-starting-style:transform-[translateY(150%)]
      [&[data-ending-style]:not([data-limited]):not([data-swipe-direction])]:transform-[translateY(150%)]
      data-ending-style:data-[swipe-direction=down]:transform-[translateY(calc(var(--toast-swipe-movement-y)+150%))]
      data-expanded:data-ending-style:data-[swipe-direction=down]:transform-[translateY(calc(var(--toast-swipe-movement-y)+150%))]
      data-ending-style:data-[swipe-direction=left]:transform-[translateX(calc(var(--toast-swipe-movement-x)-150%))_translateY(var(--offset-y))]
      data-expanded:data-ending-style:data-[swipe-direction=left]:transform-[translateX(calc(var(--toast-swipe-movement-x)-150%))_translateY(var(--offset-y))]
      data-ending-style:data-[swipe-direction=right]:transform-[translateX(calc(var(--toast-swipe-movement-x)+150%))_translateY(var(--offset-y))]
      data-expanded:data-ending-style:data-[swipe-direction=right]:transform-[translateX(calc(var(--toast-swipe-movement-x)+150%))_translateY(var(--offset-y))]
      data-ending-style:data-[swipe-direction=up]:transform-[translateY(calc(var(--toast-swipe-movement-y)-150%))]
      data-expanded:data-ending-style:data-[swipe-direction=up]:transform-[translateY(calc(var(--toast-swipe-movement-y)-150%))]
      h-(--height) data-expanded:h-(--toast-height)
      [transition:transform_0.5s_cubic-bezier(0.22,1,0.36,1),opacity_0.5s,height_0.15s]"
    >
      <Toast.Content
        className="overflow-hidden transition-opacity
      [transition-duration:250ms] data-behind:pointer-events-none
      data-behind:opacity-0 data-expanded:pointer-events-auto
      data-expanded:opacity-100"
      >
        <Toast.Title className="text-[0.975rem] leading-5 font-medium" />
        <Toast.Description
          className="text-[0.925rem] leading-5
        text-gray-700"
        />
        <Toast.Close
          className="absolute top-2 right-2 flex h-5 w-5 items-center
          justify-center rounded border-none bg-transparent text-gray-500
          hover:bg-gray-100 hover:text-gray-700"
          aria-label="Close"
        >
          <XIcon className="h-4 w-4" />
        </Toast.Close>
      </Toast.Content>
    </Toast.Root>
  ));
}

function HomeContent() {
  const [isPending, startTransition] = useTransition();
  const [isPendingJsonUpload, startTransitionJsonUpload] = useTransition();
  const {
    fpsRef,
    isLoaded,
    canvasref,
    srcVideoref,
    handleJsonUpload,
    setBG,
    start,
    stop,
    registerErrorHandler,
    unregisterErrorHandler,
  } = useTFSegmentation();
  const [jsonData, setJsonData] = useState<any>(null);
  const [isPreviewHidden, setIsPreviewHidden] = useState(true);
  const toast = Toast.useToastManager();

  const errorHandler = (error: Error) => {
    toast.add({
      title: "Error occured while starting videostream",
      description: error.message,
    });
  };

  useEffect(() => {
    registerErrorHandler(errorHandler);
    return () => {
      unregisterErrorHandler(errorHandler);
    };
  }, []);

  const handleJsonUploadLocal = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file && file.type === "application/json") {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const content = e.target?.result as string;
          startTransitionJsonUpload(async () => {
            const parsedData = JSON.parse(content);
            const result = await handleJsonUpload(parsedData);
            if (!result) {
              toast.add({
                title: "Error occured while uploading JSON",
                description: "Please check your JSON file and try again",
              });
              return;
            }
            setJsonData(parsedData);
          });
          console.log("JSON файл загружен и сохранен:", file.name);
        } catch (error) {
          console.error("Ошибка парсинга JSON:", error);
          alert("Неверный формат JSON файла!");
        }
      };
      reader.readAsText(file);
    }
  };

  const handleStartStream = () => {
    startTransition(start);
  };

  const handleStopStream = () => {
    stop();
  };
  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-4 bg-gray-900 p-8">
      <div className="h-full w-full bg-gray-800 rounded-lg overflow-hidden relative">
        {!isLoaded && (
          <button
            onClick={handleStartStream}
            className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center gap-6 w-8/10 h-48
          rounded-3xl transition-all duration-200 font-bold shadow-2xl
          hover:shadow-3xl border-2 text-3xl 
             ${
               isPending
                 ? "bg-gray-600 hover:bg-gray-600 text-gray-400 border-gray-500"
                 : "bg-blue-600 hover:bg-blue-700 text-white border-blue-500"
             }
          }`}
            disabled={isPending || isLoaded}
          >
            {!isPending ? (
              <>
                <svg
                  className="w-12 h-12"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0
                001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Запустить сегментацию
              </>
            ) : (
              <>
                <div
                  className="animate-spin rounded-full h-12 w-12 border-b-2
                border-white"
                ></div>
                Загрузка...
              </>
            )}
          </button>
        )}
        <canvas
          ref={canvasref}
          className="w-full h-full object-cover transition-opacity"
        />
        {isLoaded && (
          <p
            ref={fpsRef}
            className="absolute top-5 right-5 text-white text-sm bg-gray-500 rounded-lg p-2"
          >
            FPS: 0
          </p>
        )}
        <video
          ref={srcVideoref}
          autoPlay
          playsInline
          onClick={() => {
            setIsPreviewHidden(true);
          }}
          className={`absolute w-1/3 aspect-auto top-5 left-5 object-cover border-2 border-gray-300 rounded-lg ${
            isLoaded && !isPreviewHidden ? "block" : "hidden"
          }`}
        />

        {isPreviewHidden && isLoaded && (
          <button
            onClick={() => {
              setIsPreviewHidden(false);
            }}
            className="absolute top-5 left-5 w-15 h-15 bg-gray-500 opacity-50 text-white text-center flex items-center justify-center font-bold rounded-lg"
          >
            {!isPending ? (
              <>
                <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0
                001.555.832l3.197-2.132a1 1 0 000-1.664z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                
              </>
            ) : (
              <>
                <div
                  className="animate-spin rounded-full h-12 w-12 border-b-2
                border-white"
                ></div>
                Загрузка...
              </>
            )}
          </button>
        )}
      </div>
      <div className="flex flex-row gap-12 w-full max-w-6xl">
        {/* Левая верхняя кнопка - импорт JSON */}
        <div className="flex-1">
          <div
            className={`flex flex-col items-center justify-center w-full h-24
              bg-blue-600 text-white rounded-3xl hover:bg-blue-700 transition-all
              duration-200 font-bold cursor-pointer shadow-2xl border-2
              border-blue-500 text-lg ${
                isPendingJsonUpload || !isLoaded
                  ? "bg-gray-600 hover:bg-gray-600 text-gray-400 border-gray-500"
                  : "bg-blue-600 hover:bg-blue-700 text-white border-blue-500"
              }`}
          >
            <label className="flex items-center justify-center gap-4 w-full">
              <svg
                className="w-12 h-12"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15
              13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              Импорт JSON
              <input
                type="file"
                accept=".json"
                onChange={handleJsonUploadLocal}
                className="hidden"
                disabled={isPendingJsonUpload || !isLoaded}
              />
            </label>
            {jsonData && (
              <div className="text-center text-gray-300 text-lg font-medium">
                Файл загружен
              </div>
            )}
            {isPendingJsonUpload && (
              <div className="text-center text-gray-300 text-lg font-medium">
                Загрузка...
              </div>
            )}
          </div>
        </div>

        {/* Правая верхняя кнопка - Выбрать фон */}
        <div className="flex-1">
          <BGSelector
            uid={1}
            className="flex flex-col items-center justify-center w-full h-24
          bg-blue-600 text-white rounded-3xl shadow-2xl border-2 border-blue-500
          hover:bg-blue-700 transition-all duration-200"
            onBackgroundSelected={(bg) => {
              setBG(bg.img);
            }}
          >
            {/* Заголовок "Выбрать фон" */}
            <div className="flex items-center justify-center gap-4">
              <svg
                className="w-12 h-12"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0
                012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0
                00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span className="text-lg font-bold">Выбрать фон</span>
            </div>

            {/* Компонент выбора фона */}
          </BGSelector>
        </div>
        <div
          className={`flex-1 overflow-hidden transition-all ${
            isLoaded ? "max-w-full opacity-100" : "max-w-0 opacity-0"
          }`}
        >
          <button
            onClick={handleStopStream}
            className={`flex items-center justify-center gap-6 w-full h-24
              rounded-3xl transition-all duration-200 font-bold shadow-2xl
                  bg-red-600 hover:bg-red-700 text-white border-red-400
              hover:shadow-3xl border-2 text-lg`}
          >
            <svg
              className="w-12 h-12"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
              />
            </svg>
            Остановить сегментацию
          </button>
        </div>
      </div>
    </div>
  );
}

// Главный компонент Home должен быть последним
export default function Home() {
  return (
    <TFSegmentationProvider>
      <Toast.Provider>
        <HomeContent />
        <Toast.Portal>
          <Toast.Viewport
            className="fixed z-10 top-auto right-4 bottom-4
          mx-auto flex w-[250px] sm:right-8 sm:bottom-8 sm:w-[300px]"
          >
            <ToastList />
          </Toast.Viewport>
        </Toast.Portal>
      </Toast.Provider>
    </TFSegmentationProvider>
  );
}

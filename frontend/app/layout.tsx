import './globals.css'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html>
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>WebRTC Webcam</title>
      </head>
      <body className="h-dvh w-dvw">{children}</body>
    </html>
  )
}
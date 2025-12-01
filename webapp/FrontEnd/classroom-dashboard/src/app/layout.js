// frontend/app/layout.js
import Sidebar from './components/Sidebar'
import { AuthProvider } from './context/AuthContext'
import './globals.css'

export const metadata = {
  title: 'Classroom Analytics',
  description: 'AI-powered classroom monitoring system',
}

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <body className="bg-gray-50">
        <AuthProvider>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="flex-1 transition-all duration-300">
              {children}
            </main>
          </div>
        </AuthProvider>
      </body>
    </html>
  )
}
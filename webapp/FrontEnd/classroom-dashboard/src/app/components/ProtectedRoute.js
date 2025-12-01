// frontend/app/components/ProtectedRoute.js
'use client'
import { useRouter } from 'next/navigation'
import { useEffect } from 'react'
import { useAuth } from '../context/AuthContext'

export default function ProtectedRoute({ children }) {
    const { user, loading } = useAuth()
    const router = useRouter()

    useEffect(() => {
        if (!loading && !user) {
            router.push('/login')
        }
    }, [user, loading, router])

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
                    <p className="text-white mt-4">Đang tải...</p>
                </div>
            </div>
        )
    }

    if (!user) {
        return null
    }

    return children
}
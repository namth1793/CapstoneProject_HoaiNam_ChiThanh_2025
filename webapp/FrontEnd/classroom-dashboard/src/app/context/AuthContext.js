// frontend/app/context/AuthContext.js
'use client'
import { createContext, useContext, useEffect, useState } from 'react'

const AuthContext = createContext()

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null)
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        // Kiểm tra thông tin đăng nhập khi component mount
        const token = localStorage.getItem('access_token')
        const userData = localStorage.getItem('user')

        if (token && userData) {
            try {
                setUser(JSON.parse(userData))
            } catch (error) {
                console.error('Error parsing user data:', error)
                localStorage.removeItem('access_token')
                localStorage.removeItem('user')
            }
        }
        setLoading(false)
    }, [])

    const login = (userData, token) => {
        localStorage.setItem('access_token', token)
        localStorage.setItem('user', JSON.stringify(userData))
        setUser(userData)
    }

    const logout = () => {
        localStorage.removeItem('access_token')
        localStorage.removeItem('user')
        setUser(null)
        // Chuyển hướng đến trang login
        window.location.href = '/login'
    }

    const value = {
        user,
        login,
        logout,
        loading
    }

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    )
}

export function useAuth() {
    const context = useContext(AuthContext)
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider')
    }
    return context
}
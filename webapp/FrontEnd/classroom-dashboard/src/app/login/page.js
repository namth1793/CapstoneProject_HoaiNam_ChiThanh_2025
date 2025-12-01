// frontend/app/login/page.js
'use client'
import { useRouter } from 'next/navigation'
import { useEffect, useState } from 'react'
import { useAuth } from '../context/AuthContext'

export default function Login() {
    const [isLogin, setIsLogin] = useState(true)
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        full_name: ''
    })
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const { user, login, loading: authLoading } = useAuth()
    const router = useRouter()

    useEffect(() => {
        // Nếu đã đăng nhập, chuyển hướng đến dashboard
        if (user && !authLoading) {
            router.push('/')
        }
    }, [user, authLoading, router])

    // Hiển thị loading khi đang kiểm tra authentication
    if (authLoading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto"></div>
                    <p className="text-white mt-4">Đang tải...</p>
                </div>
            </div>
        )
    }

    // Nếu đã đăng nhập, không hiển thị trang login
    if (user) {
        return null
    }

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        })
    }

    const handleSubmit = async (e) => {
        e.preventDefault()
        setLoading(true)
        setError('')

        try {
            const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register'
            const response = await fetch(`http://localhost:8000${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData),
            })

            const data = await response.json()

            if (!response.ok) {
                throw new Error(data.detail || 'Có lỗi xảy ra')
            }

            if (isLogin) {
                // Đăng nhập thành công
                login(data.user, data.access_token)
                router.push('/')
            } else {
                // Đăng ký thành công, chuyển sang đăng nhập
                setIsLogin(true)
                setError('Đăng ký thành công! Vui lòng đăng nhập.')
                setFormData({
                    username: '',
                    email: '',
                    password: '',
                    full_name: ''
                })
            }
        } catch (error) {
            setError(error.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700 flex items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-md">
                <div className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-gray-800 mb-2">
                        Classroom Analytics
                    </h1>
                    <p className="text-gray-600">
                        {isLogin ? 'Đăng nhập vào tài khoản' : 'Tạo tài khoản mới'}
                    </p>
                </div>

                {error && (
                    <div className={`mb-6 p-4 rounded-lg ${error.includes('thành công')
                        ? 'bg-green-100 text-green-800 border border-green-200'
                        : 'bg-red-100 text-red-800 border border-red-200'
                        }`}>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    {!isLogin && (
                        <>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Họ và tên
                                </label>
                                <input
                                    type="text"
                                    name="full_name"
                                    value={formData.full_name}
                                    onChange={handleChange}
                                    required={!isLogin}
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                                    placeholder="Nhập họ và tên"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Email
                                </label>
                                <input
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required={!isLogin}
                                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                                    placeholder="Nhập email"
                                />
                            </div>
                        </>
                    )}

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Tên đăng nhập
                        </label>
                        <input
                            type="text"
                            name="username"
                            value={formData.username}
                            onChange={handleChange}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                            placeholder="Nhập tên đăng nhập"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Mật khẩu
                        </label>
                        <input
                            type="password"
                            name="password"
                            value={formData.password}
                            onChange={handleChange}
                            required
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition duration-200"
                            placeholder="Nhập mật khẩu"
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 hover:cursor-pointer hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-semibold transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? 'Đang xử lý...' : (isLogin ? 'Đăng nhập' : 'Đăng ký')}
                    </button>
                </form>

                <div className="mt-6 text-center">
                    <button
                        onClick={() => {
                            setIsLogin(!isLogin)
                            setError('')
                            setFormData({
                                username: '',
                                email: '',
                                password: '',
                                full_name: ''
                            })
                        }}
                        className="text-blue-600 hover:text-blue-800 font-medium hover:cursor-pointer"
                    >
                        {isLogin
                            ? 'Chưa có tài khoản? Đăng ký ngay'
                            : 'Đã có tài khoản? Đăng nhập'
                        }
                    </button>
                </div>

                {isLogin && (
                    <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                        <p className="text-sm text-gray-600 text-center">
                            <strong>Tài khoản demo:</strong> admin / admin123
                        </p>
                    </div>
                )}
            </div>
        </div>
    )
}
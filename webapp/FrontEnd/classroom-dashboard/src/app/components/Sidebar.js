// frontend/app/components/Sidebar.js
'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useAuth } from '../context/AuthContext'

export default function Sidebar() {
    const { user, logout } = useAuth()
    const pathname = usePathname()

    // Ẩn sidebar khi đang ở trang login hoặc khi chưa đăng nhập
    if (pathname === '/login' || !user) {
        return null
    }

    const menuItems = [
        { href: '/', label: 'Dashboard', icon: '📊' },
        { href: '/live-class', label: 'Live Class', icon: '🎥' },
        { href: '/attendance', label: 'Attendance', icon: '👥' },
        { href: '/analytics', label: 'Analytics', icon: '📈' },
        { href: '/reports', label: 'Reports', icon: '📝' },
    ]

    return (
        <div className="w-64 sidebar min-h-screen left-0 top-0">
            {/* Header Sidebar */}
            <div className="p-6 border-b border-gray-700">
                <h1 className="text-xl font-bold">Classroom Analytics</h1>
            </div>

            {/* User Info */}
            <div className="p-4 border-b border-gray-700">
                <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                        <span className="font-bold">
                            {user.full_name?.charAt(0) || user.username?.charAt(0)}
                        </span>
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-white truncate">
                            {user.full_name || user.username}
                        </p>
                        <p className="text-xs truncate">@{user.username}</p>
                    </div>
                </div>
            </div>

            {/* Navigation Menu */}
            <nav className="p-4">
                <ul className="space-y-2">
                    {menuItems.map((item) => (
                        <li key={item.href}>
                            <Link
                                href={item.href}
                                className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition duration-200 ${pathname === item.href
                                    ? 'bg-blue-600 text-white'
                                    : 'hover:bg-gray-800 hover:text-white'
                                    }`}
                            >
                                <span className="text-lg">{item.icon}</span>
                                <span className="font-medium">{item.label}</span>
                            </Link>
                        </li>
                    ))}
                </ul>
            </nav>

            {/* Logout Button */}
            <div className="absolute bottom-4 left-4 right-4 ml-13">
                <button
                    onClick={logout}
                    className="flex items-center space-x-3 px-4 py-3 hover:cursor-pointer hover:bg-gray-800 hover:text-white rounded-lg transition duration-200"
                >
                    <span className="font-medium">Đăng xuất</span>
                </button>
            </div>
        </div>
    )
}
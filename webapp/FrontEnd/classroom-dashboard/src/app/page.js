// frontend/app/page.js
'use client'
import { useEffect, useState } from 'react'
import BehaviorDistribution from './components/BehaviorDistribution'
import DashboardStats from './components/DashboardStats'
import EmotionChart from './components/EmotionChart'
import EngagementDistribution from './components/EngagementDistribution'
import ProtectedRoute from './components/ProtectedRoute'
import { useAuth } from './context/AuthContext'

export default function Dashboard() {
  const { user, loading: authLoading } = useAuth()
  const [stats, setStats] = useState(null)
  const [students, setStudents] = useState([])
  const [dataLoading, setDataLoading] = useState(true)

  const fetchData = async () => {
    try {
      const token = localStorage.getItem('access_token')

      const [statsRes, studentsRes] = await Promise.all([
        fetch('http://localhost:8000/api/dashboard/stats', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }),
        fetch('http://localhost:8000/api/students', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })
      ])

      if (statsRes.status === 401 || studentsRes.status === 401) {
        // Token hết hạn, đăng xuất
        localStorage.removeItem('access_token')
        localStorage.removeItem('user')
        window.location.href = '/login'
        return
      }

      const statsData = await statsRes.json()
      const studentsData = await studentsRes.json()

      setStats(statsData)
      setStudents(studentsData.students || [])
      setDataLoading(false)
    } catch (error) {
      console.error('Error fetching data:', error)
      setDataLoading(false)
    }
  }

  useEffect(() => {
    if (user) {
      fetchData()
      const interval = setInterval(fetchData, 10000) // Refresh every 10 seconds
      return () => clearInterval(interval)
    }
  }, [user])

  return (
    <ProtectedRoute>
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white">Dashboard</h1>
              <p className="text-white">Chào mừng trở lại, {user?.full_name || user?.username}!</p>
            </div>
            <div className="text-sm text-gray-500">
              Cập nhật: {new Date().toLocaleTimeString('vi-VN')}
            </div>
          </div>

          {dataLoading ? (
            <div className="flex justify-center items-center min-h-64">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            <>
              <DashboardStats stats={stats} />

              {/* Grid layout cho 3 biểu đồ chính */}
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 mt-6">
                <div className="xl:col-span-1">
                  <EmotionChart />
                </div>

                <div className="xl:col-span-1">
                  <BehaviorDistribution />
                </div>

                <div className="xl:col-span-1">
                  <EngagementDistribution students={students} />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </ProtectedRoute>
  )
}
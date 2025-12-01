// src/app/analytics/page.js
'use client'
import { useEffect, useState } from 'react'

export default function AnalyticsPage() {
    const [emotionTrend, setEmotionTrend] = useState([])
    const [engagementData, setEngagementData] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        fetchAnalyticsData()
    }, [])

    const fetchAnalyticsData = async () => {
        try {
            setError(null)
            const [emotionRes, engagementRes] = await Promise.all([
                fetch('http://localhost:8000/api/analytics/emotion-trend'),
                fetch('http://localhost:8000/api/analytics/engagement')
            ])

            if (!emotionRes.ok || !engagementRes.ok) {
                throw new Error('Failed to fetch analytics data')
            }

            const emotionData = await emotionRes.json()
            const engagementData = await engagementRes.json()

            setEmotionTrend(emotionData.emotion_trend || [])
            setEngagementData(engagementData.engagement_data || [])
        } catch (error) {
            console.error('Error fetching analytics:', error)
            setError('Failed to load analytics data')
        } finally {
            setLoading(false)
        }
    }

    // Calculate statistics
    const avgEngagement = engagementData.length > 0
        ? engagementData.reduce((sum, item) => sum + item.engagement, 0) / engagementData.length
        : 0

    const getEngagementLevel = (engagement) => {
        if (engagement >= 80) return { level: 'High', color: 'text-green-600', bg: 'bg-green-100', border: 'border-green-200' }
        if (engagement >= 60) return { level: 'Medium', color: 'text-yellow-600', bg: 'bg-yellow-100', border: 'border-yellow-200' }
        return { level: 'Low', color: 'text-red-600', bg: 'bg-red-100', border: 'border-red-200' }
    }

    const getEmotionColor = (emotion) => {
        const colors = {
            happy: { text: 'text-green-600', bg: 'bg-green-100', border: 'border-green-200' },
            neutral: { text: 'text-blue-600', bg: 'bg-blue-100', border: 'border-blue-200' },
            sad: { text: 'text-red-600', bg: 'bg-red-100', border: 'border-red-200' },
            surprise: { text: 'text-purple-600', bg: 'bg-purple-100', border: 'border-purple-200' },
            angry: { text: 'text-orange-600', bg: 'bg-orange-100', border: 'border-orange-200' }
        }
        return colors[emotion] || colors.neutral
    }

    if (loading) {
        return (
            <div className="analytics-page flex justify-center items-center min-h-screen">
                <div className="loading-spinner"></div>
                <span className="ml-3 text-lg text-white">Loading Analytics...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className="analytics-page flex justify-center items-center min-h-screen">
                <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full mx-4">
                    <div className="text-red-500 text-center mb-4">
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 text-center mb-2">Connection Error</h3>
                    <p className="text-gray-600 text-center mb-6">{error}</p>
                    <button
                        onClick={fetchAnalyticsData}
                        className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition duration-200"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="analytics-page bg-black p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Class Analytics</h1>
                    <p className="text-blue-100 text-lg">Deep insights into student performance and engagement</p>
                </div>

                {/* Key Metrics Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    {/* Average Engagement */}
                    <div className="bg-white rounded-2xl shadow-xl p-6 text-center">
                        <div className="w-16 h-16 bg-gradient-to-r from-green-400 to-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-gray-800 mb-2">Average Engagement</h3>
                        <div className="text-3xl font-bold text-green-600 mb-2">{avgEngagement.toFixed(1)}%</div>
                        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getEngagementLevel(avgEngagement).bg} ${getEngagementLevel(avgEngagement).color} ${getEngagementLevel(avgEngagement).border}`}>
                            {getEngagementLevel(avgEngagement).level}
                        </div>
                    </div>

                    {/* Top Emotion */}
                    <div className="bg-white rounded-2xl shadow-xl p-6 text-center">
                        <div className="w-16 h-16 bg-gradient-to-r from-purple-400 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-gray-800 mb-2">Dominant Emotion</h3>
                        <div className="text-3xl font-bold text-purple-600 mb-2 capitalize">
                            {emotionTrend.length > 0
                                ? Object.entries(emotionTrend[emotionTrend.length - 1])
                                    .filter(([key]) => !['time'].includes(key))
                                    .reduce((a, b) => a[1] > b[1] ? a : b)[0]
                                : 'N/A'
                            }
                        </div>
                        <div className="text-gray-600 text-sm">Current class mood</div>
                    </div>

                    {/* Active Students */}
                    <div className="bg-white rounded-2xl shadow-xl p-6 text-center">
                        <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                            </svg>
                        </div>
                        <h3 className="text-lg font-semibold text-gray-800 mb-2">Active Analysis</h3>
                        <div className="text-3xl font-bold text-blue-600 mb-2">{engagementData.length}</div>
                        <div className="text-gray-600 text-sm">Students Tracked</div>
                    </div>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    {/* Emotion Trend Chart */}
                    <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                        <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6">
                            <h2 className="text-2xl font-bold text-white flex items-center">
                                <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                                Emotion Trend Analysis
                            </h2>
                        </div>
                        <div className="p-6">
                            <div className="overflow-x-auto">
                                <table className="w-full min-w-full">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="text-left p-4 text-gray-700 font-semibold">Time</th>
                                            <th className="text-center p-4 text-gray-700 font-semibold">😊 Happy</th>
                                            <th className="text-center p-4 text-gray-700 font-semibold">😐 Neutral</th>
                                            <th className="text-center p-4 text-gray-700 font-semibold">😢 Sad</th>
                                            <th className="text-center p-4 text-gray-700 font-semibold">😲 Surprise</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {emotionTrend.map((trend, index) => (
                                            <tr key={index} className="hover:bg-gray-50 transition duration-150">
                                                <td className="p-4">
                                                    <div className="flex items-center">
                                                        <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-semibold text-sm mr-3">
                                                            {trend.time}
                                                        </div>
                                                        <span className="text-gray-800 font-medium">{trend.time}</span>
                                                    </div>
                                                </td>
                                                <td className="p-4 text-center">
                                                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-green-100 text-green-800 border border-green-200">
                                                        {trend.happy}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-center">
                                                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-800 border border-blue-200">
                                                        {trend.neutral}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-center">
                                                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-red-100 text-red-800 border border-red-200">
                                                        {trend.sad}
                                                    </span>
                                                </td>
                                                <td className="p-4 text-center">
                                                    <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-purple-100 text-purple-800 border border-purple-200">
                                                        {trend.surprise}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    {/* Engagement Ranking */}
                    <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                        <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
                            <h2 className="text-2xl font-bold text-white flex items-center">
                                <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                                </svg>
                                Student Engagement Ranking
                            </h2>
                        </div>
                        <div className="p-6">
                            <div className="space-y-4">
                                {engagementData
                                    .sort((a, b) => b.engagement - a.engagement)
                                    .map((student, index) => {
                                        const engagementInfo = getEngagementLevel(student.engagement)
                                        return (
                                            <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition duration-150">
                                                <div className="flex items-center">
                                                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-sm mr-4 ${index === 0 ? 'bg-yellow-500' :
                                                        index === 1 ? 'bg-gray-400' :
                                                            index === 2 ? 'bg-orange-500' : 'bg-blue-500'
                                                        }`}>
                                                        #{index + 1}
                                                    </div>
                                                    <div>
                                                        <div className="font-semibold text-gray-800">{student.student}</div>
                                                        <div className="text-sm text-gray-600">Engagement Level</div>
                                                    </div>
                                                </div>
                                                <div className="text-right">
                                                    <div className={`text-xl font-bold ${engagementInfo.color}`}>
                                                        {student.engagement}%
                                                    </div>
                                                    <div className={`text-sm font-medium ${engagementInfo.color}`}>
                                                        {engagementInfo.level}
                                                    </div>
                                                </div>
                                            </div>
                                        )
                                    })
                                }
                            </div>

                            {/* Engagement Summary */}
                            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                                <h3 className="font-semibold text-gray-800 mb-3">Engagement Summary</h3>
                                <div className="grid grid-cols-3 gap-4 text-center">
                                    <div>
                                        <div className="text-2xl font-bold text-green-600">
                                            {engagementData.filter(s => s.engagement >= 80).length}
                                        </div>
                                        <div className="text-sm text-green-800 font-medium">High</div>
                                    </div>
                                    <div>
                                        <div className="text-2xl font-bold text-yellow-600">
                                            {engagementData.filter(s => s.engagement >= 60 && s.engagement < 80).length}
                                        </div>
                                        <div className="text-sm text-yellow-800 font-medium">Medium</div>
                                    </div>
                                    <div>
                                        <div className="text-2xl font-bold text-red-600">
                                            {engagementData.filter(s => s.engagement < 60).length}
                                        </div>
                                        <div className="text-sm text-red-800 font-medium">Low</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
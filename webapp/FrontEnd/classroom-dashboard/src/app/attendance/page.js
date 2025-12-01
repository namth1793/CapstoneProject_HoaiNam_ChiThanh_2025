// src/app/attendance/page.js
'use client'
import { useEffect, useState } from 'react'

export default function AttendancePage() {
    const [attendanceData, setAttendanceData] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        fetchAttendanceData()
    }, [])

    const fetchAttendanceData = async () => {
        try {
            setError(null)
            const response = await fetch('http://localhost:8000/api/attendance')

            if (!response.ok) {
                throw new Error('Failed to fetch attendance data')
            }

            const data = await response.json()
            setAttendanceData(data.attendance || [])
        } catch (error) {
            console.error('Error fetching attendance:', error)
            setError('Failed to load attendance data')
        } finally {
            setLoading(false)
        }
    }

    // Calculate statistics
    const presentCount = attendanceData.filter(item => item.status === 'present').length
    const absentCount = attendanceData.filter(item => item.status === 'absent').length
    const totalStudents = attendanceData.length
    const presentPercentage = totalStudents > 0 ? Math.round((presentCount / totalStudents) * 100) : 0
    const absentPercentage = totalStudents > 0 ? Math.round((absentCount / totalStudents) * 100) : 0

    if (loading) {
        return (
            <div className="attendance-page flex justify-center items-center min-h-screen">
                <div className="loading-spinner"></div>
                <span className="ml-3 text-lg text-white">Loading Attendance...</span>
            </div>
        )
    }

    if (error) {
        return (
            <div className="attendance-page flex justify-center items-center min-h-screen">
                <div className="bg-white rounded-2xl shadow-xl p-8 max-w-md w-full mx-4">
                    <div className="text-red-500 text-center mb-4">
                        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                    </div>
                    <h3 className="text-xl font-bold text-gray-800 text-center mb-2">Connection Error</h3>
                    <p className="text-gray-600 text-center mb-6">{error}</p>
                    <button
                        onClick={fetchAttendanceData}
                        className="w-full bg-blue-600 text-white py-3 rounded-xl font-semibold hover:bg-blue-700 transition duration-200"
                    >
                        Try Again
                    </button>
                </div>
            </div>
        )
    }

    return (
        <div className="attendance-page bg-black p-6 min-h-screen">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-white mb-3">Attendance Management</h1>
                    <p className="text-blue-100 text-lg">Track and manage student attendance in real-time</p>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                    {/* Today's Attendance Card */}
                    <div className="xl:col-span-2">
                        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
                                <h2 className="text-2xl font-bold text-white flex items-center">
                                    <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    Today's Attendance
                                </h2>
                                <p className="text-blue-100 mt-1">{new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                            </div>

                            <div className="p-6">
                                <div className="overflow-hidden rounded-xl border border-gray-200">
                                    <table className="w-full">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                <th className="text-left p-4 text-gray-700 font-semibold">Student</th>
                                                <th className="text-left p-4 text-gray-700 font-semibold">Time In</th>
                                                <th className="text-center p-4 text-gray-700 font-semibold">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-200">
                                            {attendanceData.map((record, index) => (
                                                <tr key={index} className="hover:bg-gray-50 transition duration-150">
                                                    <td className="p-4">
                                                        <div className="flex items-center">
                                                            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-semibold text-sm mr-3">
                                                                {record.student.split(' ').map(n => n[0]).join('')}
                                                            </div>
                                                            <span className="text-gray-800 font-medium">{record.student}</span>
                                                        </div>
                                                    </td>
                                                    <td className="p-4">
                                                        <span className={`text-lg font-semibold ${record.time_in === '-' ? 'text-gray-400' : 'text-gray-700'}`}>
                                                            {record.time_in}
                                                        </span>
                                                    </td>
                                                    <td className="p-4 text-center">
                                                        <span className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold ${record.status === 'present'
                                                            ? 'bg-green-100 text-green-800 border border-green-200'
                                                            : 'bg-red-100 text-red-800 border border-red-200'
                                                            }`}>
                                                            {record.status === 'present' ? (
                                                                <>
                                                                    <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                                                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                                    </svg>
                                                                    Present
                                                                </>
                                                            ) : (
                                                                <>
                                                                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                                                    </svg>
                                                                    Absent
                                                                </>
                                                            )}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Attendance Overview Card */}
                    <div className="xl:col-span-1">
                        <div className="bg-white rounded-2xl shadow-xl overflow-hidden h-full">
                            <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
                                <h2 className="text-2xl font-bold text-white flex items-center">
                                    <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                    </svg>
                                    Attendance Overview
                                </h2>
                            </div>

                            <div className="p-6">
                                {/* Progress Bars */}
                                <div className="space-y-6 mb-8">
                                    {/* Present Progress */}
                                    <div>
                                        <div className="flex justify-between items-center mb-3">
                                            <div className="flex items-center">
                                                <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                                                <span className="text-gray-700 font-semibold">Present</span>
                                            </div>
                                            <span className="text-2xl font-bold text-green-600">{presentPercentage}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-4">
                                            <div
                                                className="bg-green-500 h-4 rounded-full transition-all duration-500 ease-out"
                                                style={{ width: `${presentPercentage}%` }}
                                            ></div>
                                        </div>
                                        <div className="text-right mt-1">
                                            <span className="text-sm text-gray-500">{presentCount} students</span>
                                        </div>
                                    </div>

                                    {/* Absent Progress */}
                                    <div>
                                        <div className="flex justify-between items-center mb-3">
                                            <div className="flex items-center">
                                                <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                                                <span className="text-gray-700 font-semibold">Absent</span>
                                            </div>
                                            <span className="text-2xl font-bold text-red-600">{absentPercentage}%</span>
                                        </div>
                                        <div className="w-full bg-gray-200 rounded-full h-4">
                                            <div
                                                className="bg-red-500 h-4 rounded-full transition-all duration-500 ease-out"
                                                style={{ width: `${absentPercentage}%` }}
                                            ></div>
                                        </div>
                                        <div className="text-right mt-1">
                                            <span className="text-sm text-gray-500">{absentCount} students</span>
                                        </div>
                                    </div>
                                </div>

                                {/* Summary Stats */}
                                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-100">
                                    <h3 className="text-lg font-semibold text-gray-800 text-center mb-4">Today's Summary</h3>
                                    <div className="grid grid-cols-3 gap-4 text-center">
                                        <div>
                                            <div className="text-3xl font-bold text-blue-600 mb-1">{totalStudents}</div>
                                            <div className="text-sm text-blue-800 font-medium">Total Students</div>
                                        </div>
                                        <div>
                                            <div className="text-3xl font-bold text-green-600 mb-1">{presentCount}</div>
                                            <div className="text-sm text-green-800 font-medium">Present</div>
                                        </div>
                                        <div>
                                            <div className="text-3xl font-bold text-red-600 mb-1">{absentCount}</div>
                                            <div className="text-sm text-red-800 font-medium">Absent</div>
                                        </div>
                                    </div>
                                </div>

                                {/* Additional Info */}
                                <div className="mt-6 p-4 bg-yellow-50 rounded-xl border border-yellow-200">
                                    <div className="flex items-start">
                                        <svg className="w-5 h-5 text-yellow-600 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                        </svg>
                                        <div>
                                            <p className="text-sm text-yellow-800">
                                                <strong>Note:</strong> Attendance data updates in real-time. Last updated {new Date().toLocaleTimeString()}
                                            </p>
                                        </div>
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
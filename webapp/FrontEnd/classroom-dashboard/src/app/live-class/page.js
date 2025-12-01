// src/app/live-class/page.js
'use client'
import { useEffect, useRef, useState } from 'react';

export default function LiveClass() {
    const [liveData, setLiveData] = useState(null)
    const [isConnected, setIsConnected] = useState(false)
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [cameraError, setCameraError] = useState('')
    const videoRef = useRef(null)
    const streamRef = useRef(null)

    useEffect(() => {
        // WebSocket connection for live data
        const ws = new WebSocket('ws://localhost:8000/ws/live')

        ws.onopen = () => {
            console.log('WebSocket connected')
            setIsConnected(true)
        }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            setLiveData(data)
        }

        ws.onclose = () => {
            console.log('WebSocket disconnected')
            setIsConnected(false)
        }

        return () => {
            ws.close()
            stopCamera()
        }
    }, [])

    // SỬA: Không tự động bật camera khi component mount
    // useEffect(() => {
    //     startCamera()
    // }, [])

    // Hàm bật camera - SỬA LẠI
    const startCamera = async () => {
        console.log('Attempting to start camera...')
        try {
            setCameraError('')
            console.log('Requesting camera access...')

            // SỬA: Kiểm tra xem trình duyệt có hỗ trợ mediaDevices không
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Trình duyệt không hỗ trợ truy cập camera')
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            })

            console.log('Camera access granted, stream:', stream)

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                // SỬA QUAN TRỌNG: Đảm bảo video element được cấu hình đúng
                videoRef.current.autoplay = true
                videoRef.current.playsInline = true
                videoRef.current.muted = true

                streamRef.current = stream
                setIsCameraOn(true)
                console.log('Camera started successfully')
            }
        } catch (error) {
            console.error('Error accessing camera:', error)
            setCameraError(`Không thể truy cập camera: ${error.message}`)
            setIsCameraOn(false)
        }
    }

    // Hàm tắt camera
    const stopCamera = () => {
        console.log('Stopping camera...')
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => {
                track.stop()
                console.log('Stopped track:', track.kind)
            })
            streamRef.current = null
        }
        if (videoRef.current) {
            videoRef.current.srcObject = null
        }
        setIsCameraOn(false)
        setCameraError('')
        console.log('Camera stopped')
    }

    // Hàm chuyển đổi camera
    const toggleCamera = () => {
        if (isCameraOn) {
            stopCamera()
        } else {
            startCamera()
        }
    }

    // SỬA: Thêm hàm kiểm tra và khắc phục camera
    const fixCameraDisplay = () => {
        if (videoRef.current && streamRef.current) {
            // Thử load lại video element
            videoRef.current.load()
            console.log('Attempting to fix camera display...')
        }
    }

    // SỬA: Thêm useEffect để xử lý khi camera state thay đổi
    useEffect(() => {
        if (isCameraOn && videoRef.current) {
            // Đảm bảo video element có đúng attributes
            videoRef.current.autoplay = true
            videoRef.current.playsInline = true
            videoRef.current.muted = true
        }
    }, [isCameraOn])

    const getEmotionColor = (emotion) => {
        const colors = {
            happy: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-200', emoji: '😊' },
            neutral: { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-200', emoji: '😐' },
            sad: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-200', emoji: '😢' },
            surprise: { bg: 'bg-purple-100', text: 'text-purple-800', border: 'border-purple-200', emoji: '😲' },
            angry: { bg: 'bg-orange-100', text: 'text-orange-800', border: 'border-orange-200', emoji: '😠' }
        }
        return colors[emotion] || colors.neutral
    }

    const getEngagementColor = (engagement) => {
        if (engagement >= 80) return 'text-green-400'
        if (engagement >= 60) return 'text-yellow-400'
        return 'text-red-400'
    }

    return (
        <div className="min-h-screen bg-black p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-4xl font-bold text-white mb-2">Live Classroom</h1>
                        <p className="text-blue-200">Real-time classroom monitoring and analytics</p>
                    </div>
                    <div className="flex items-center space-x-4">
                        <button
                            onClick={toggleCamera}
                            className={`flex items-center px-4 py-2 rounded-lg font-semibold transition duration-200 ${isCameraOn
                                ? 'bg-red-600 hover:bg-red-700 text-white'
                                : 'bg-green-600 hover:bg-green-700 text-white'
                                }`}
                        >
                            {isCameraOn ? (
                                <>
                                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                    </svg>
                                    Turn Off Camera
                                </>
                            ) : (
                                <>
                                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                                    </svg>
                                    Turn On Camera
                                </>
                            )}
                        </button>
                        <div className="flex items-center space-x-3">
                            <div className={`w-4 h-4 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                            <span className="text-white font-medium">
                                {isConnected ? 'Live Connected' : 'Disconnected'}
                            </span>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                    {/* Main Video Feed */}
                    <div className="xl:col-span-3">
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl shadow-2xl overflow-hidden border border-gray-700">
                            <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6">
                                <div className="flex items-center justify-between">
                                    <h2 className="text-2xl font-bold text-white flex items-center">
                                        <svg className="w-6 h-6 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                        </svg>
                                        Live Camera Feed
                                    </h2>
                                    <div className="flex items-center space-x-2">
                                        <div className={`w-3 h-3 rounded-full ${isCameraOn ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                                        <span className="text-blue-100 text-sm">
                                            {isCameraOn ? 'Camera is On' : 'Camera is Off'}
                                        </span>
                                    </div>
                                </div>
                            </div>

                            <div className="aspect-video bg-black flex items-center justify-center relative overflow-hidden">
                                {isCameraOn ? (
                                    <>
                                        <video
                                            ref={videoRef}
                                            autoPlay
                                            playsInline
                                            muted
                                            className="w-full h-full object-cover"
                                            onLoadedMetadata={() => console.log('Video metadata loaded')}
                                            onCanPlay={() => console.log('Video can play')}
                                            onError={(e) => console.error('Video error:', e)}
                                        />

                                        {/* SỬA: Thêm nút khắc phục khi camera không hiển thị */}
                                        {cameraError && (
                                            <div className="absolute inset-0 bg-black bg-opacity-80 flex items-center justify-center">
                                                <div className="text-center text-white p-6">
                                                    <div className="text-4xl mb-4">⚠️</div>
                                                    <p className="text-xl font-semibold mb-2">Camera Error</p>
                                                    <p className="text-gray-300 mb-4">{cameraError}</p>
                                                    <div className="flex space-x-3 justify-center">
                                                        <button
                                                            onClick={startCamera}
                                                            className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition duration-200"
                                                        >
                                                            Thử lại
                                                        </button>
                                                        <button
                                                            onClick={fixCameraDisplay}
                                                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition duration-200"
                                                        >
                                                            Khắc phục
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Simulated bounding boxes for demo */}
                                        {liveData && liveData.students && liveData.students.filter(s => s.status === 'present').map((student, index) => {
                                            const emotion = getEmotionColor(student.emotion)
                                            return (
                                                <div
                                                    key={student.id}
                                                    className="absolute border-2 rounded-xl p-3 backdrop-blur-sm bg-black bg-opacity-30 transition-all duration-300 hover:scale-105"
                                                    style={{
                                                        left: `${15 + index * 20}%`,
                                                        top: `${25 + index * 15}%`,
                                                        width: '18%',
                                                        height: '25%',
                                                        borderColor: emotion.border.replace('border-', '').split('-')[1] + '500'
                                                    }}
                                                >
                                                    <div className="flex items-center justify-between mb-2">
                                                        <div className="text-white text-sm font-semibold truncate">
                                                            {student.name.split(' ').pop()}
                                                        </div>
                                                        <span className="text-lg">{emotion.emoji}</span>
                                                    </div>
                                                    <div className={`text-xs px-2 py-1 rounded-full ${emotion.bg} ${emotion.text} font-medium capitalize`}>
                                                        {student.emotion}
                                                    </div>
                                                    <div className="text-white text-xs mt-2">
                                                        Eng: <span className={getEngagementColor(student.engagement * 100)}>
                                                            {Math.round(student.engagement * 100)}%
                                                        </span>
                                                    </div>
                                                </div>
                                            )
                                        })}
                                    </>
                                ) : (
                                    <div className="text-center text-white p-8">
                                        <div className="text-6xl mb-4">📹</div>
                                        <p className="text-xl font-semibold mb-4">Camera Feed</p>
                                        <p className="text-gray-400 mb-6">
                                            {cameraError || 'Nhấn "Bật Camera" để bắt đầu phát trực tiếp'}
                                        </p>
                                        <button
                                            onClick={startCamera}
                                            className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-semibold transition duration-200 flex items-center mx-auto"
                                        >
                                            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                                            </svg>
                                            Turn On Camera
                                        </button>
                                    </div>
                                )}

                                {/* Camera Controls */}
                                {isCameraOn && (
                                    <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-4">
                                        <button
                                            onClick={stopCamera}
                                            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-semibold transition duration-200 flex items-center"
                                        >
                                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                                            </svg>
                                            Dừng Camera
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Student Grid - Giữ nguyên */}
                        {/* ... rest of the code remains the same ... */}
                    </div>

                    {/* Live Stats Sidebar - Giữ nguyên */}
                    {/* ... rest of the code remains the same ... */}
                </div>
            </div>
        </div>
    )
}
// frontend/app/components/DashboardStats.js
export default function DashboardStats({ stats }) {
    const statCards = [
        {
            title: 'Total Students',
            value: stats.total_students,
            color: 'bg-blue-500',
            icon: '👥'
        },
        {
            title: 'Present',
            value: stats.present_count,
            color: 'bg-green-500',
            icon: '✅'
        },
        {
            title: 'Absent',
            value: stats.absent_count,
            color: 'bg-red-500',
            icon: '❌'
        },
        {
            title: 'Attendance Rate',
            value: `${stats.attendance_rate}%`,
            color: 'bg-purple-500',
            icon: '📊'
        },
        {
            title: 'Avg Engagement',
            value: `${stats.avg_engagement}%`,
            color: 'bg-orange-500',
            icon: '🎯'
        },
        {
            title: 'Dominant Emotion',
            value: stats.current_emotion,
            color: 'bg-pink-500',
            icon: '😊'
        }
    ]

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {statCards.map((card, index) => (
                <div key={index} className="bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-center space-x-3">
                        <div className={`${card.color} w-12 h-12 rounded-lg flex items-center justify-center text-white`}>
                            <span className="text-xl">{card.icon}</span>
                        </div>
                        <div>
                            <h3 className="text-sm font-medium text-gray-600">{card.title}</h3>
                            <p className="text-lg font-semibold text-gray-900">{card.value}</p>
                        </div>
                    </div>
                </div>
            ))}
        </div>
    )
}
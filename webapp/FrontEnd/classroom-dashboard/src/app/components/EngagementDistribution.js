// frontend/components/EngagementDistribution.js
import { Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const EngagementDistribution = ({ students }) => {
    const getEngagementDistribution = () => {
        const ranges = [
            { range: '0-20%', count: 0, color: '#EF4444' },
            { range: '21-40%', count: 0, color: '#F59E0B' },
            { range: '41-60%', count: 0, color: '#EAB308' },
            { range: '61-80%', count: 0, color: '#84CC16' },
            { range: '81-100%', count: 0, color: '#10B981' }
        ]

        students.forEach(student => {
            const engagement = student.engagement || 0
            if (engagement <= 20) ranges[0].count++
            else if (engagement <= 40) ranges[1].count++
            else if (engagement <= 60) ranges[2].count++
            else if (engagement <= 80) ranges[3].count++
            else ranges[4].count++
        })

        return ranges
    }

    const engagementData = getEngagementDistribution()

    return (
        <div className="bg-white rounded-xl p-6 h-full">
            <h2 className="text-xl font-semibold text-black mb-4">Engagement Distribution</h2>
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={engagementData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
                        <XAxis
                            dataKey="range"
                            stroke="#9CA3AF"
                            fontSize={12}
                        />
                        <YAxis
                            stroke="#9CA3AF"
                            fontSize={12}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1F2937',
                                border: 'none',
                                borderRadius: '8px',
                                color: 'white'
                            }}
                            itemStyle={{ color: 'white' }}
                        />
                        <Legend />
                        <Bar
                            dataKey="count"
                            name="Number of Students"
                            radius={[4, 4, 0, 0]}
                        >
                            {engagementData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default EngagementDistribution
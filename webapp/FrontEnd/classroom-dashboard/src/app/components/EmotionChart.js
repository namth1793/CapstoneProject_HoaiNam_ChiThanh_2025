// frontend/components/EmotionChart.js
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

const EmotionChart = () => {
    const emotionData = [
        { name: 'Happy', value: 35 },
        { name: 'Neutral', value: 45 },
        { name: 'Sad', value: 10 },
        { name: 'Surprise', value: 10 }
    ]

    const COLORS = ['#10B981', '#6B7280', '#EF4444', '#F59E0B']

    return (
        <div className="bg-white rounded-xl p-6 h-full">
            <h2 className="text-xl font-semibold text-black mb-4">Emotion Distribution</h2>
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={emotionData}
                            cx="50%"
                            cy="50%"
                            innerRadius={40}
                            outerRadius={70}
                            paddingAngle={2}
                            dataKey="value"
                        >
                            {emotionData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Pie>
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#1F2937',
                                border: 'none',
                                borderRadius: '8px',
                                color: 'white'
                            }}
                            itemStyle={{ color: 'white' }}
                        />
                        <Legend
                            wrapperStyle={{
                                fontSize: '12px',
                                color: '#9CA3AF'
                            }}
                        />
                    </PieChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default EmotionChart
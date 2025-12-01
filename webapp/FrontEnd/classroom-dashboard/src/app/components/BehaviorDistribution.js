// frontend/components/BehaviorDistribution.js
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const BehaviorDistribution = () => {
    const behaviorData = [
        { name: 'Paying\nAttention', value: 65, fill: '#10B981' },
        { name: 'Distracted', value: 15, fill: '#EF4444' },
        { name: 'Active', value: 12, fill: '#3B82F6' },
        { name: 'Inactive', value: 8, fill: '#6B7280' }
    ]

    return (
        <div className="bg-white rounded-xl p-6 h-full">
            <h2 className="text-xl font-semibold text-black mb-4">Behavior Distribution</h2>
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={behaviorData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#4B5563" />
                        <XAxis
                            dataKey="name"
                            stroke="#9CA3AF"
                            fontSize={12}
                            interval={0}
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
                        <Bar
                            dataKey="value"
                            name="Percentage"
                            radius={[4, 4, 0, 0]}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

export default BehaviorDistribution
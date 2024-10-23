const express = require('express')
const app = express()

const cors = require('cors')
app.use(cors())
app.use(express.json())
const db = require('./db/db')
db()




const authRoutes = require('./routes/authRoutes')
app.use('/auth',authRoutes)

app.get('/ping',(req, res)=>{
    res.send('PONG')
})

app.listen(process.env.PORT, ()=>{
    console.log(`Server running at port ${process.env.PORT}`)
})
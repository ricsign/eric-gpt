import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/springs.css'
import './styles/tokens.css'
import './styles/base.css'
import { App } from './App'
import { StoreProvider } from './state/store'

const root = document.getElementById('root')
if (!root) throw new Error('#root is missing from index.html')

createRoot(root).render(
  <StrictMode>
    <StoreProvider>
      <App />
    </StoreProvider>
  </StrictMode>,
)

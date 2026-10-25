import { useEffect, useState } from 'react'
import Landing from './Landing.jsx'
import Workspace from './Workspace.jsx'

// Hash routing keeps this a static site: "#/app" is the workspace, anything else the landing page.
const isApp = () => window.location.hash.startsWith('#/app')

export default function App() {
  const [app, setApp] = useState(isApp())
  useEffect(() => {
    const onHash = () => {
      setApp(isApp())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  return app ? <Workspace /> : <Landing />
}

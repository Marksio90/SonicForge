import type { Metadata } from "next";
import Script from "next/script";
import "./globals.css";

export const metadata: Metadata = {
  title: "SonicForge — Dashboard",
  description: "AI-Powered 24/7 Music Radio Platform — Control Panel",
  manifest: "/manifest.json",
  themeColor: "#6366f1",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "SonicForge",
  },
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/icon-192.png" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
      </head>
      <body className="min-h-screen bg-sf-bg text-gray-200 antialiased">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 p-6 overflow-auto">{children}</main>
        </div>
        
        {/* Service Worker Registration */}
        <Script id="register-sw" strategy="afterInteractive">
          {`
            if ('serviceWorker' in navigator) {
              window.addEventListener('load', () => {
                navigator.serviceWorker.register('/sw.js')
                  .then(registration => {
                    console.log('SW registered:', registration);
                  })
                  .catch(error => {
                    console.log('SW registration failed:', error);
                  });
              });
            }
          `}
        </Script>
      </body>
    </html>
  );
}

function Sidebar() {
  const navItems = [
    { label: "Dashboard", href: "/", icon: "⚡" },
    { label: "Live", href: "/live", icon: "🔴", highlight: true },
    { label: "Pipeline", href: "/pipeline", icon: "🔄" },
    { label: "Tracks", href: "/tracks", icon: "🎵" },
    { label: "Stream", href: "/stream", icon: "📡" },
    { label: "Schedule", href: "/schedule", icon: "📅" },
    { label: "Analytics", href: "/analytics", icon: "📊" },
    { label: "Agents", href: "/agents", icon: "🤖" },
    { label: "Visuals", href: "/visuals", icon: "🎨" },
    { label: "Settings", href: "/settings", icon: "⚙️" },
  ];

  return (
    <aside className="w-64 bg-sf-surface border-r border-sf-border flex flex-col">
      <div className="p-6 border-b border-sf-border">
        <h1 className="text-2xl font-bold text-white">
          <span className="text-sf-accent">Sonic</span>Forge
        </h1>
        <p className="text-xs text-gray-500 mt-1">AI Music Radio Platform</p>
      </div>
      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => (
          <a
            key={item.href}
            href={item.href}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-400 hover:text-white hover:bg-sf-border/50 transition-colors"
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </a>
        ))}
      </nav>
      <div className="p-4 border-t border-sf-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-sf-green animate-pulse" />
          <span className="text-xs text-gray-500">System Operational</span>
        </div>
      </div>
    </aside>
  );
}

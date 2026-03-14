import type { Metadata } from "next";
import { Inter, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Pricing Actuarial | ML vs GLM",
  description:
    "Dashboard interactivo de tarificacion actuarial: comparacion de modelos GLM, GBM y LightGBM para pricing de seguros de auto.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es" className={`${inter.variable} ${playfair.variable}`}>
      <body className="bg-paper text-ink antialiased dark:bg-[#1A1A1A] dark:text-[#FAFAF8]">
        {children}
      </body>
    </html>
  );
}

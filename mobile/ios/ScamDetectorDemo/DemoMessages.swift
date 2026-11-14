import Foundation

/// Synthetic inbox for the demo. Messages are hand-written examples in the style of real
/// Pakistani SMS (from data/probes/handwritten_v1.csv); numbers and links are placeholders.
/// `expectedScam` is the true label, so the demo can show where the model is right and wrong.
struct DemoMessage: Identifiable, Hashable {
    let id = UUID()
    let sender: String
    let time: String
    let text: String
    let expectedScam: Bool
}

enum DemoInbox {
    static let messages: [DemoMessage] = [
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "9:41",
                    text: "Apka HBL ka account suspicious activity ki waja se block ho gaya hai. Abhi OTP bataen 03XX-XXXXXXX par warna account hamesha k liye band ho jaye ga.",
                    expectedScam: true),
        DemoMessage(sender: "JazzCash", time: "9:30",
                    text: "1234 is your JazzCash OTP. Do not share it with anyone.",
                    expectedScam: false),
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "9:12",
                    text: "Mubarak ho! Aap ne 50,000 Rs inaam jeeta hai. Hasil karne ke liye 03XX-XXXXXXX par call karein.",
                    expectedScam: true),
        DemoMessage(sender: "Ammi", time: "8:55",
                    text: "Ammi ne kaha hai shaam ko jaldi ghar aa jana, khana bana hua hai",
                    expectedScam: false),
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "8:20",
                    text: "BISP: Aap ki 25000 ki qist aa gayi hai. Wasool karne ke liye apna CNIC aur OTP is number par bhejein.",
                    expectedScam: true),
        DemoMessage(sender: "MeezanBank", time: "Yesterday",
                    text: "Apke Meezan Bank account mein 15,000 Rs deposit hue hain. Available balance 42,300 Rs.",
                    expectedScam: false),
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "Yesterday",
                    text: "FBR ki taraf se apka Easypaisa acc freeze kr dia gya ha. Unblock krwane k lie abhi OTP sms karin.",
                    expectedScam: true),
        DemoMessage(sender: "IESCO", time: "Yesterday",
                    text: "Apka bijli ka bill 4,500 Rs hai. Due date 20 tareekh hai. Shukriya - IESCO",
                    expectedScam: false),
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "Mon",
                    text: "PTA notice: aap ki SIM 24 ghante mein band ho jayegi. Biometric verification ke liye 03XX-XXXXXXX par call karein.",
                    expectedScam: true),
        DemoMessage(sender: "8070", time: "Mon",
                    text: "Dear customer, Rs 5,000 has been transferred to your JazzCash wallet. TID 12345678",
                    expectedScam: false),
        DemoMessage(sender: "+92 3XX XXXXXXX", time: "Sun",
                    text: "Your Easypaisa account will be suspended today. Share the 4-digit code sent to you to keep it active.",
                    expectedScam: true),
        DemoMessage(sender: "Daraz", time: "Sun",
                    text: "Apka order Daraz se dispatch ho gaya hai aur 2-3 din mein deliver hoga.",
                    expectedScam: false),
    ]

    /// Quick examples for the Check tab.
    static let samples: [String] = [
        "Aap ka JazzCash account block ho gaya hai, foran verify karein",
        "Mubarak ho! Aap ki BISP raqam 25000 Rs mansoor ho gayi hai, OTP batayein",
        "Your HBL a/c credited with Rs 25,000 salary",
        "Bhai kal milte hain",
    ]
}
